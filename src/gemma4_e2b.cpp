// gemma4_e2b.cpp — CrispASR runtime for Google Gemma-4-E2B
//
// Architecture: USM Conformer audio encoder (12L, 1024d, chunked attention,
// LightConv1d, macaron FFN) + Gemma4 LLM decoder (35L, 1536d, GQA 8Q/1KV,
// per-layer embeddings, hybrid sliding/full attention, SwiGLU).
//
// Audio: 128-bin log-mel at 16kHz, 40ms frames, max 30s.
// Tokenizer: BPE (262K vocab) stored in GGUF.

#include "gemma4_e2b.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/greedy_decode.h"
#include "core/mel.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// ── Hyperparameters ─────────────────────────────────────────────────────────

struct g4e_audio_hp {
    uint32_t hidden_size = 1024;
    uint32_t num_layers = 12;
    uint32_t num_heads = 8;
    uint32_t head_dim = 128; // hidden / heads
    uint32_t conv_kernel_size = 5;
    uint32_t chunk_size = 12;
    uint32_t context_left = 13;
    uint32_t output_proj_dims = 1536;
    float residual_weight = 0.5f;
    float attention_logit_cap = 50.0f;
};

struct g4e_llm_hp {
    uint32_t hidden_size = 1536;
    uint32_t num_layers = 35;
    uint32_t num_heads = 8;
    uint32_t num_kv_heads = 1;
    uint32_t head_dim = 256;
    uint32_t intermediate_size = 6144; // layers 0-14; layers 15+ use 12288
    uint32_t vocab_size = 262144;
    uint32_t max_position_embeddings = 131072;
    uint32_t sliding_window = 512;
    float rope_theta = 10000.0f;
    float final_logit_softcapping = 30.0f;
    float rms_norm_eps = 1e-6f;
};

// ── Model tensors ───────────────────────────────────────────────────────────

struct g4e_audio_layer {
    // Macaron FFN 1
    ggml_tensor* ffn1_pre_ln = nullptr;
    ggml_tensor* ffn1_up_w = nullptr;
    ggml_tensor* ffn1_down_w = nullptr;
    ggml_tensor* ffn1_post_ln = nullptr;

    // Self-attention
    ggml_tensor* attn_pre_ln = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_per_dim_scale = nullptr; // [head_dim]
    ggml_tensor* attn_rel_k_w = nullptr;       // [hidden, hidden] — relative position bias
    ggml_tensor* attn_post_ln = nullptr;

    // LightConv1d
    ggml_tensor* conv_pre_ln = nullptr;
    ggml_tensor* conv_gate_w = nullptr; // [2*hidden, hidden] — GLU gating
    ggml_tensor* conv_dw_w = nullptr;   // [hidden, 1, k] — depthwise conv
    ggml_tensor* conv_ln = nullptr;
    ggml_tensor* conv_out_w = nullptr; // [hidden, hidden]

    // Macaron FFN 2
    ggml_tensor* ffn2_pre_ln = nullptr;
    ggml_tensor* ffn2_up_w = nullptr;
    ggml_tensor* ffn2_down_w = nullptr;
    ggml_tensor* ffn2_post_ln = nullptr;

    // Output norm
    ggml_tensor* out_ln = nullptr;
};

struct g4e_llm_layer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* q_proj = nullptr; // [n_heads*head_dim, hidden]
    ggml_tensor* k_proj = nullptr; // [kv_heads*head_dim, hidden]
    ggml_tensor* v_proj = nullptr;
    ggml_tensor* o_proj = nullptr; // [hidden, n_heads*head_dim]
    ggml_tensor* q_norm = nullptr; // [head_dim]
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* post_attn_norm = nullptr;
    ggml_tensor* pre_ffn_norm = nullptr;
    ggml_tensor* gate_proj = nullptr;
    ggml_tensor* up_proj = nullptr;
    ggml_tensor* down_proj = nullptr;
    ggml_tensor* post_ffn_norm = nullptr;
    // Per-layer embeddings (PLE)
    ggml_tensor* ple_gate = nullptr; // [256, hidden]
    ggml_tensor* ple_proj = nullptr; // [hidden, 256]
    ggml_tensor* post_ple_norm = nullptr;
    ggml_tensor* layer_scalar = nullptr; // [1]
};

struct g4e_model {
    g4e_audio_hp audio_hp;
    g4e_llm_hp llm_hp;

    // Weight context
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Mel resources (stored in GGUF, preferred over runtime generation)
    ggml_tensor* mel_window = nullptr;  // [n_fft] Hann window
    ggml_tensor* mel_filters = nullptr; // [n_freqs, n_mels] filterbank

    // Audio subsampling
    ggml_tensor* sub_conv0_w = nullptr;      // [128, 1, 3, 3]
    ggml_tensor* sub_norm0_w = nullptr;      // [128]
    ggml_tensor* sub_conv1_w = nullptr;      // [32, 128, 3, 3]
    ggml_tensor* sub_norm1_w = nullptr;      // [32]
    ggml_tensor* sub_input_proj_w = nullptr; // [1024, 1024]

    // Audio conformer layers
    std::vector<g4e_audio_layer> audio_layers;

    // Audio output projection
    ggml_tensor* audio_output_proj_w = nullptr; // [1536, 1024]
    ggml_tensor* audio_output_proj_b = nullptr; // [1536]

    // Audio embedding projection (post-conformer → LLM hidden)
    ggml_tensor* audio_embed_proj_w = nullptr; // [1536, 1536]

    // LLM embeddings
    ggml_tensor* llm_embed_w = nullptr;    // [vocab, hidden]
    ggml_tensor* llm_ple_w = nullptr;      // [vocab, 35*256] — per-layer embeddings
    ggml_tensor* llm_final_norm = nullptr; // [hidden]

    // LLM layers
    std::vector<g4e_llm_layer> llm_layers;

    // Tokenizer
    std::vector<std::string> vocab;
    std::vector<std::string> merges;
};

// ── Context ─────────────────────────────────────────────────────────────────

struct gemma4_e2b_context {
    g4e_model model;
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    int n_threads = 4;
    int verbosity = 1;
    float temperature = 0.0f;

    std::string model_path;
    std::vector<uint8_t> compute_meta; // scratch for graph building

    // KV cache for LLM decode
    ggml_context* kv_ctx = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    int kv_max_ctx = 0;

    // Pre-computed mel resources (generated at init, not stored in GGUF)
    std::vector<float> mel_window;     // Hann window [n_fft]
    std::vector<float> mel_filterbank; // [n_freqs * n_mels]

    // Special token IDs (looked up at init from vocab)
    int bos_id = 2;
    int eos_id = 1;
    int start_of_turn_id = -1;
    int end_of_turn_id = -1;
};

// ── Conformer encoder graph builder ─────────────────────────────────────────
// Builds a single ggml graph for the full 12-layer USM Conformer encoder.
// Input: mel features [n_mels, T_mel] after Conv2D subsampling → [hidden, T_sub]
// Output: [output_proj_dims, T_sub]

static ggml_tensor* build_macaron_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pre_ln, ggml_tensor* up_w,
                                      ggml_tensor* down_w, ggml_tensor* post_ln, float residual_weight, float eps) {
    // Half-step FFN: x + residual_weight * post_ln(down(silu(up(pre_ln(x)))))
    ggml_tensor* h = ggml_rms_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, pre_ln);
    h = ggml_mul_mat(ctx, up_w, h);
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, down_w, h);
    ggml_tensor* normed = ggml_rms_norm(ctx, h, eps);
    normed = ggml_mul(ctx, normed, post_ln);
    return ggml_add(ctx, x, ggml_scale(ctx, normed, residual_weight));
}

static ggml_tensor* build_light_conv1d(ggml_context* ctx, ggml_tensor* x, const g4e_audio_layer& L, int T, int hidden,
                                       float eps) {
    // LightConv1d: pre_ln → gate_proj (GLU: split → sigmoid gate) → depthwise_conv1d(causal, k=5) →
    // conv_norm → out_proj + residual
    ggml_tensor* residual = x;
    ggml_tensor* h = ggml_rms_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, L.conv_pre_ln);

    // GLU gating: gate_proj produces [2*hidden, T], split into value + gate
    h = ggml_mul_mat(ctx, L.conv_gate_w, h); // [2*hidden, T]
    int half = hidden;
    ggml_tensor* val = ggml_view_2d(ctx, h, half, T, h->nb[1], 0);
    ggml_tensor* gate = ggml_view_2d(ctx, h, half, T, h->nb[1], half * ggml_type_size(h->type));
    h = ggml_mul(ctx, val, ggml_sigmoid(ctx, gate)); // [hidden, T]

    // Causal depthwise conv1d (kernel=5, left-pad=4)
    // Transpose [hidden, T] → [T, hidden] for conv_1d_dw
    ggml_tensor* ht = ggml_cont(ctx, ggml_transpose(ctx, h));
    int k = 5;
    int pad_left = k - 1; // causal: all padding on left
    ht = ggml_pad_ext(ctx, ht, pad_left, 0, 0, 0, 0, 0, 0, 0);
    ht = ggml_conv_1d_dw(ctx, L.conv_dw_w, ht, 1, 0, 1);
    if (ggml_n_dims(ht) > 2)
        ht = ggml_reshape_2d(ctx, ht, ht->ne[0], ht->ne[1]);
    h = ggml_cont(ctx, ggml_transpose(ctx, ht)); // back to [hidden, T]

    // Conv norm + out projection
    h = ggml_rms_norm(ctx, h, eps);
    h = ggml_mul(ctx, h, L.conv_ln);
    h = ggml_mul_mat(ctx, L.conv_out_w, h);

    return ggml_add(ctx, residual, h);
}

// ── FFT (Cooley-Tukey + DFT fallback for non-power-of-2) ───────────────────
// Handles n_fft=400 (= 2^4 × 25) by recursing down to a 25-point DFT.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void g4e_dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

static void g4e_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        g4e_dft(in, N, out);
        return;
    }
    float* even = in + N;
    for (int i = 0; i < half_N; i++)
        even[i] = in[2 * i];
    float* even_fft = out + 2 * N;
    g4e_fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; i++)
        odd[i] = in[2 * i + 1];
    float* odd_fft = even_fft + N;
    g4e_fft(odd, half_N, odd_fft);

    for (int k = 0; k < half_N; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float re = std::cos(ang);
        float im = std::sin(ang);
        float re_odd = odd_fft[2 * k];
        float im_odd = odd_fft[2 * k + 1];
        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half_N)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

static void g4e_fft_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    g4e_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

// ── Mel filterbank generation ──────────────────────────────────────────────
// HTK mel scale (same as Whisper/HF WhisperFeatureExtractor).

static void g4e_gen_mel_filterbank(int n_mels, int n_fft, int sr, std::vector<float>& fb) {
    const int n_freqs = n_fft / 2 + 1;
    fb.assign((size_t)n_freqs * n_mels, 0.0f);

    auto hz_to_mel = [](double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); };
    auto mel_to_hz = [](double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); };

    double mel_lo = hz_to_mel(0.0);
    double mel_hi = hz_to_mel(sr / 2.0);
    std::vector<double> mel_pts((size_t)(n_mels + 2));
    for (int i = 0; i < n_mels + 2; i++)
        mel_pts[i] = mel_to_hz(mel_lo + (mel_hi - mel_lo) * i / (n_mels + 1));

    std::vector<double> fft_freqs((size_t)n_freqs);
    for (int i = 0; i < n_freqs; i++)
        fft_freqs[i] = (double)sr * i / n_fft;

    // Triangular filters with slaney normalization (2 / (f_hi - f_lo))
    for (int m = 0; m < n_mels; m++) {
        double lo = mel_pts[m], ctr = mel_pts[m + 1], hi = mel_pts[m + 2];
        double enorm = (hi > lo) ? 2.0 / (hi - lo) : 0.0;
        for (int f = 0; f < n_freqs; f++) {
            double val = 0.0;
            if (fft_freqs[f] >= lo && fft_freqs[f] <= ctr && ctr > lo)
                val = (fft_freqs[f] - lo) / (ctr - lo);
            else if (fft_freqs[f] > ctr && fft_freqs[f] <= hi && hi > ctr)
                val = (hi - fft_freqs[f]) / (hi - ctr);
            // fb layout: (n_freqs, n_mels) for FbLayout::FreqsMels
            fb[(size_t)f * n_mels + m] = (float)(val * enorm);
        }
    }
}

static void g4e_gen_hann_window(int n_fft, std::vector<float>& win) {
    win.resize(n_fft);
    for (int i = 0; i < n_fft; i++)
        win[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / n_fft));
}

// ── KV cache init ──────────────────────────────────────────────────────────

static bool g4e_kv_init(gemma4_e2b_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    if (ctx->kv_k)
        return true;

    const auto& lhp = ctx->model.llm_hp;
    const int hd = (int)lhp.head_dim;
    const int n_kv = (int)lhp.num_kv_heads;
    const int n_lay = (int)lhp.num_layers;

    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "gemma4_e2b: failed to alloc kv buffer\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    ctx->kv_max_ctx = max_ctx;

    if (ctx->verbosity >= 1)
        fprintf(stderr, "gemma4_e2b: kv cache %d MiB (hd=%d max_ctx=%d n_kv=%d n_lay=%d)\n",
                (int)((kbytes + vbytes) / 1048576), hd, max_ctx, n_kv, n_lay);
    return true;
}

// ── Conformer self-attention (full, no chunking) ───────────────────────────
// First-pass: uses full attention with per_dim_scale. Chunked attention
// and relative position bias to be added when differential testing passes.

static ggml_tensor* build_conformer_self_attn(ggml_context* ctx, ggml_tensor* x, const g4e_audio_layer& L,
                                              const g4e_audio_hp& hp) {
    const int hd = (int)hp.head_dim;
    const int n_h = (int)hp.num_heads;
    const int T = (int)x->ne[1];
    const float eps = 1e-6f;

    ggml_tensor* residual = x;
    ggml_tensor* h = ggml_rms_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, L.attn_pre_ln);

    ggml_tensor* Q = ggml_mul_mat(ctx, L.attn_q_w, h); // [n_h*hd, T]
    ggml_tensor* K = ggml_mul_mat(ctx, L.attn_k_w, h);
    ggml_tensor* V = ggml_mul_mat(ctx, L.attn_v_w, h);

    Q = ggml_reshape_3d(ctx, Q, hd, n_h, T);
    K = ggml_reshape_3d(ctx, K, hd, n_h, T);
    V = ggml_reshape_3d(ctx, V, hd, n_h, T);

    // Per-dim scale: multiply each Q dimension by the learned scale
    // per_dim_scale is [head_dim], broadcast across heads and time
    if (L.attn_per_dim_scale) {
        // Reshape scale to [hd, 1, 1] for broadcast
        ggml_tensor* scale = ggml_reshape_3d(ctx, L.attn_per_dim_scale, hd, 1, 1);
        Q = ggml_mul(ctx, Q, scale);
    }

    // Permute to flash-attention layout: (hd, T, n_h)
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Full bidirectional attention (no mask), with logit softcapping
    float attn_scale = 1.0f; // per_dim_scale replaces 1/sqrt(d)
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, attn_scale, 0.0f, hp.attention_logit_cap);
    attn = ggml_reshape_2d(ctx, attn, hd * n_h, T);

    // Output projection
    attn = ggml_mul_mat(ctx, L.attn_o_w, attn);

    // Post-attention norm + residual
    attn = ggml_rms_norm(ctx, attn, eps);
    attn = ggml_mul(ctx, attn, L.attn_post_ln);
    return ggml_add(ctx, residual, attn);
}

// ── LLM graph builder (KV-cached) ─────────────────────────────────────────
// Builds a graph for the Gemma4 LLM with KV cache.
// Handles both prefill (T > 1) and decode (T = 1).

static ggml_cgraph* g4e_build_graph_llm_kv(gemma4_e2b_context* ctx, int n_past, int n_tokens) {
    const auto& m = ctx->model;
    const auto& lhp = m.llm_hp;
    const int d = (int)lhp.hidden_size;
    const int n_q = (int)lhp.num_heads;
    const int n_kv = (int)lhp.num_kv_heads;
    const int hd = (int)lhp.head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = lhp.rms_norm_eps;
    const float theta = lhp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;
    const int ple_dim = 256; // per-layer embedding dimension

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // PLE input: token IDs for per-layer embedding lookup
    ggml_tensor* ple_ids = nullptr;
    if (m.llm_ple_w) {
        ple_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_name(ple_ids, "ple_ids");
        ggml_set_input(ple_ids);
    }

    // Causal mask (only for prefill T > 1)
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;

    // PLE lookup: get all per-layer embeddings at once
    ggml_tensor* ple_all = nullptr;
    if (m.llm_ple_w && ple_ids) {
        // ple_w: [n_layers*ple_dim, vocab] in ggml → get_rows returns [n_layers*ple_dim, T]
        ple_all = ggml_get_rows(ctx0, m.llm_ple_w, ple_ids);
    }

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)lhp.max_position_embeddings,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    for (uint32_t il = 0; il < lhp.num_layers; il++) {
        const auto& b = m.llm_layers[il];

        // ── PLE (per-layer embedding adjustment) ──
        // Gemma4 flow (per transformers/models/gemma4/modeling_gemma4.py):
        //   gate  = act_fn(per_layer_input_gate(hidden))   # 1536 → 256
        //   gated = gate * per_layer_emb_for_this_layer    # element-wise (256,)
        //   delta = per_layer_projection(gated)            # 256 → 1536
        //   hidden += post_per_layer_input_norm(delta)
        //
        // ple_gate.weight  PyTorch (256, 1536)  → ggml ne[0]=1536, ne[1]=256.  Linear(1536→256).
        // ple_proj.weight  PyTorch (1536, 256)  → ggml ne[0]=256, ne[1]=1536.  Linear(256→1536).
        //
        // PLAN #50 — the attention block below still trips
        // ggml_can_mul_mat on full-attention layers (head_dim=512)
        // because `kvp` is built once with the local head_dim=256.
        // Fix is to read `layer_full_mask` from the GGUF and rebuild
        // kvp per layer (sliding=256, full=global_head_dim=512), with
        // matching RoPE theta + partial_rotary_factor.
        if (ple_all && b.ple_gate && b.ple_proj) {
            // Slice this layer's PLE: [ple_dim, T] from [n_layers*ple_dim, T]
            ggml_tensor* ple_slice = ggml_view_2d(ctx0, ple_all, ple_dim, T, ple_all->nb[1],
                                                  (size_t)il * ple_dim * ggml_type_size(ple_all->type));
            // hidden → 256
            ggml_tensor* gate = ggml_mul_mat(ctx0, b.ple_gate, cur);
            gate = ggml_gelu(ctx0, gate); // Gemma3n act_fn = GELU (pytorch_tanh)
            // 256 × 256 element-wise
            ggml_tensor* gated = ggml_mul(ctx0, gate, ple_slice);
            // 256 → hidden
            ggml_tensor* delta = ggml_mul_mat(ctx0, b.ple_proj, gated);
            if (b.post_ple_norm) {
                delta = ggml_rms_norm(ctx0, delta, eps);
                delta = ggml_mul(ctx0, delta, b.post_ple_norm);
            }
            cur = ggml_add(ctx0, cur, delta);
        }

        // ── Attention ──
        ggml_tensor* residual = cur;
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm);

        ggml_tensor* attn =
            core_attn::kv_self_attn(ctx0, gf, x, b.q_proj, b.k_proj, b.v_proj, b.o_proj, b.q_norm, b.k_norm, positions,
                                    (T == 1) ? nullptr : causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp);

        // Post-attention norm
        if (b.post_attn_norm) {
            attn = ggml_rms_norm(ctx0, attn, eps);
            attn = ggml_mul(ctx0, attn, b.post_attn_norm);
        }

        // Residual with optional layer_scalar
        if (b.layer_scalar)
            attn = ggml_mul(ctx0, attn, b.layer_scalar);
        cur = ggml_add(ctx0, residual, attn);

        // ── FFN (SwiGLU) ──
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        if (b.pre_ffn_norm)
            x = ggml_mul(ctx0, x, b.pre_ffn_norm);

        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.gate_proj, b.up_proj, b.down_proj);

        // Post-FFN norm
        if (b.post_ffn_norm) {
            mlp = ggml_rms_norm(ctx0, mlp, eps);
            mlp = ggml_mul(ctx0, mlp, b.post_ffn_norm);
        }

        if (b.layer_scalar)
            mlp = ggml_mul(ctx0, mlp, b.layer_scalar);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final norm
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm_final_norm);

    // Last-token-only lm_head for decode
    if (T > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);

    // lm_head = tied embed weights
    cur = ggml_mul_mat(ctx0, m.llm_embed_w, cur);

    // Logit softcapping: tanh(logits / cap) * cap
    if (lhp.final_logit_softcapping > 0.0f) {
        float cap = lhp.final_logit_softcapping;
        cur = ggml_scale(ctx0, cur, 1.0f / cap);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, cap);
    }

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ── Token embedding graph ──────────────────────────────────────────────────

static ggml_cgraph* g4e_build_graph_embed(gemma4_e2b_context* ctx, int n_tokens) {
    const int d = (int)ctx->model.llm_hp.hidden_size;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);

    ggml_tensor* emb = ggml_get_rows(ctx0, ctx->model.llm_embed_w, ids);

    // Gemma embedding scale: multiply by sqrt(hidden_size)
    emb = ggml_scale(ctx0, emb, std::sqrt((float)d));

    ggml_set_name(emb, "embeds");
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);
    ggml_free(ctx0);
    return gf;
}

// ── Run LLM with KV cache ─────────────────────────────────────────────────

static float* g4e_run_llm_kv(gemma4_e2b_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                             int* /*out_n_tokens*/, int* /*out_vocab_size*/) {
    if (!ctx || !inputs_embeds || n_tokens <= 0 || !ctx->kv_k)
        return nullptr;

    const auto& lhp = ctx->model.llm_hp;
    const int d = (int)lhp.hidden_size;
    const int vocab = (int)lhp.vocab_size;
    const int Lk = n_past + n_tokens;

    if (Lk > ctx->kv_max_ctx) {
        fprintf(stderr, "gemma4_e2b: kv overflow (%d + %d > %d)\n", n_past, n_tokens, ctx->kv_max_ctx);
        return nullptr;
    }

    // Positions
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    // Causal mask (F16, only for prefill)
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++)
            for (int k = n_past + q + 1; k < Lk; k++)
                mask[(size_t)q * Lk + k] = neginf_h;
    }

    // PLE token IDs: for decode steps we need the last token ID.
    // The caller provides embeddings, not token IDs. For the decode step,
    // we store the last-generated token ID in a field. For prefill, the
    // IDs are set by the caller via the ple_ids input tensor.
    // TODO: pass token IDs through for PLE in decode path

    ggml_cgraph* gf = g4e_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "gemma4_e2b: failed to alloc llm_kv graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gemma4_e2b: llm_kv graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out)
        return nullptr;

    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)vocab * sizeof(float));
    return result;
}

// ── Embed tokens ───────────────────────────────────────────────────────────

static float* g4e_embed_tokens(gemma4_e2b_context* ctx, const int32_t* ids, int n) {
    if (!ctx || !ids || n <= 0)
        return nullptr;
    const int d = (int)ctx->model.llm_hp.hidden_size;

    ggml_cgraph* gf = g4e_build_graph_embed(ctx, n);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, ids, 0, (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* result = (float*)malloc((size_t)d * n * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)d * n * sizeof(float));
    return result;
}

// ── Public API ──────────────────────────────────────────────────────────────

extern "C" struct gemma4_e2b_context_params gemma4_e2b_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/true, /*temperature=*/0.0f};
}

static uint32_t g4e_gguf_u32(gguf_context* ctx, const char* key, uint32_t def = 0) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_u32(ctx, id) : def;
}
static float g4e_gguf_f32(gguf_context* ctx, const char* key, float def = 0.0f) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}

extern "C" struct gemma4_e2b_context* gemma4_e2b_init_from_file(const char* path_model,
                                                                struct gemma4_e2b_context_params params) {
    auto* ctx = new gemma4_e2b_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->temperature = params.temperature;
    auto& m = ctx->model;

    // ── Read GGUF metadata ──────────────────────────────────────────────
    struct gguf_init_params gp = {true, &m.ctx_w};
    gguf_context* gctx = gguf_init_from_file(path_model, gp);
    if (!gctx) {
        fprintf(stderr, "gemma4_e2b: failed to open '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    auto& ahp = m.audio_hp;
    ahp.hidden_size = g4e_gguf_u32(gctx, "gemma4e2b.audio.hidden_size", 1024);
    ahp.num_layers = g4e_gguf_u32(gctx, "gemma4e2b.audio.num_layers", 12);
    ahp.num_heads = g4e_gguf_u32(gctx, "gemma4e2b.audio.num_heads", 8);
    ahp.head_dim = ahp.hidden_size / ahp.num_heads;
    ahp.conv_kernel_size = g4e_gguf_u32(gctx, "gemma4e2b.audio.conv_kernel_size", 5);
    ahp.chunk_size = g4e_gguf_u32(gctx, "gemma4e2b.audio.chunk_size", 12);
    ahp.context_left = g4e_gguf_u32(gctx, "gemma4e2b.audio.context_left", 13);
    ahp.output_proj_dims = g4e_gguf_u32(gctx, "gemma4e2b.audio.output_proj_dims", 1536);
    ahp.residual_weight = g4e_gguf_f32(gctx, "gemma4e2b.audio.residual_weight", 0.5f);
    ahp.attention_logit_cap = g4e_gguf_f32(gctx, "gemma4e2b.audio.attention_logit_cap", 50.0f);

    auto& lhp = m.llm_hp;
    lhp.hidden_size = g4e_gguf_u32(gctx, "gemma4e2b.llm.hidden_size", 1536);
    lhp.num_layers = g4e_gguf_u32(gctx, "gemma4e2b.llm.num_layers", 35);
    lhp.num_heads = g4e_gguf_u32(gctx, "gemma4e2b.llm.num_heads", 8);
    lhp.num_kv_heads = g4e_gguf_u32(gctx, "gemma4e2b.llm.num_kv_heads", 1);
    lhp.head_dim = g4e_gguf_u32(gctx, "gemma4e2b.llm.head_dim", 256);
    lhp.intermediate_size = g4e_gguf_u32(gctx, "gemma4e2b.llm.intermediate_size", 6144);
    lhp.vocab_size = g4e_gguf_u32(gctx, "gemma4e2b.llm.vocab_size", 262144);
    lhp.max_position_embeddings = g4e_gguf_u32(gctx, "gemma4e2b.llm.max_position_embeddings", 131072);
    lhp.sliding_window = g4e_gguf_u32(gctx, "gemma4e2b.llm.sliding_window", 512);
    lhp.rope_theta = g4e_gguf_f32(gctx, "gemma4e2b.llm.rope_theta", 10000.0f);
    lhp.final_logit_softcapping = g4e_gguf_f32(gctx, "gemma4e2b.llm.final_logit_softcapping", 30.0f);
    lhp.rms_norm_eps = g4e_gguf_f32(gctx, "gemma4e2b.llm.rms_norm_eps", 1e-6f);

    // Read tokenizer from GGUF
    int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = gguf_get_arr_n(gctx, tok_key);
        m.vocab.resize(n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, tok_key, i);
            if (s)
                m.vocab[i] = s;
        }
    }
    gguf_free(gctx);

    if (ahp.hidden_size == 0 || lhp.vocab_size == 0) {
        fprintf(stderr, "gemma4_e2b: invalid model metadata\n");
        delete ctx;
        return nullptr;
    }

    // ── Load weights ────────────────────────────────────────────────────
    // Use CPU for weights — the LLM decoder can use ggml_backend_sched
    // to auto-copy to GPU for batched matmuls, and the audio conformer
    // uses single-graph execution.
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "gemma4_e2b: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    // Load to CPU (same pattern as firered — allows native Q4_K SIMD for decoder)
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend_cpu, "gemma4_e2b", wl)) {
        fprintf(stderr, "gemma4_e2b: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    m.ctx_w = wl.ctx;
    m.buf_w = wl.buf;
    auto& ts = wl.tensors;

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = ts.find(name);
        if (it == ts.end()) {
            if (params.verbosity >= 2)
                fprintf(stderr, "gemma4_e2b: tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };

    // ── Bind audio tensors ──────────────────────────────────────────────
    m.mel_window = get("audio.mel_window");
    m.mel_filters = get("audio.mel_filters");
    m.sub_conv0_w = get("audio.subsample.conv0.weight");
    m.sub_norm0_w = get("audio.subsample.norm0.weight");
    m.sub_conv1_w = get("audio.subsample.conv1.weight");
    m.sub_norm1_w = get("audio.subsample.norm1.weight");
    m.sub_input_proj_w = get("audio.subsample.input_proj.weight");

    m.audio_layers.resize(ahp.num_layers);
    for (uint32_t i = 0; i < ahp.num_layers; i++) {
        char buf[128];
        auto& L = m.audio_layers[i];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "audio.layers.%u.%s", i, suffix);
            return get(buf);
        };
        // Macaron FFN 1
        L.ffn1_pre_ln = g("ffn1.pre_ln.weight");
        L.ffn1_up_w = g("ffn1.up.weight");
        L.ffn1_down_w = g("ffn1.down.weight");
        L.ffn1_post_ln = g("ffn1.post_ln.weight");
        // Self-attention
        L.attn_pre_ln = g("attn_pre_ln.weight");
        L.attn_q_w = g("attn.q.weight");
        L.attn_k_w = g("attn.k.weight");
        L.attn_v_w = g("attn.v.weight");
        L.attn_o_w = g("attn.o.weight");
        L.attn_per_dim_scale = g("attn.per_dim_scale");
        L.attn_rel_k_w = g("attn.rel_k.weight");
        L.attn_post_ln = g("attn_post_ln.weight");
        // LightConv1d
        L.conv_pre_ln = g("conv.pre_ln.weight");
        L.conv_gate_w = g("conv.gate_proj.weight");
        L.conv_dw_w = g("conv.dw_conv.weight");
        L.conv_ln = g("conv.conv_ln.weight");
        L.conv_out_w = g("conv.out_proj.weight");
        // Macaron FFN 2
        L.ffn2_pre_ln = g("ffn2.pre_ln.weight");
        L.ffn2_up_w = g("ffn2.up.weight");
        L.ffn2_down_w = g("ffn2.down.weight");
        L.ffn2_post_ln = g("ffn2.post_ln.weight");
        // Output norm
        L.out_ln = g("out_ln.weight");
    }
    m.audio_output_proj_w = get("audio.output_proj.weight");
    m.audio_output_proj_b = get("audio.output_proj.bias");
    m.audio_embed_proj_w = get("audio.embed_proj.weight");

    // ── Bind LLM tensors ────────────────────────────────────────────────
    m.llm_embed_w = get("llm.embed_tokens.weight");
    m.llm_ple_w = get("llm.embed_tokens_per_layer.weight");
    m.llm_final_norm = get("llm.norm.weight");

    m.llm_layers.resize(lhp.num_layers);
    for (uint32_t i = 0; i < lhp.num_layers; i++) {
        char buf[128];
        auto& L = m.llm_layers[i];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.layers.%u.%s", i, suffix);
            return get(buf);
        };
        L.attn_norm = g("attn_norm.weight");
        L.q_proj = g("attn.q_proj.weight");
        L.k_proj = g("attn.k_proj.weight");
        L.v_proj = g("attn.v_proj.weight");
        L.o_proj = g("attn.o_proj.weight");
        L.q_norm = g("attn.q_norm.weight");
        L.k_norm = g("attn.k_norm.weight");
        L.post_attn_norm = g("post_attn_norm.weight");
        L.pre_ffn_norm = g("pre_ffn_norm.weight");
        L.gate_proj = g("ffn.gate.weight");
        L.up_proj = g("ffn.up.weight");
        L.down_proj = g("ffn.down.weight");
        L.post_ffn_norm = g("post_ffn_norm.weight");
        L.ple_gate = g("ple_gate.weight");
        L.ple_proj = g("ple_proj.weight");
        L.post_ple_norm = g("post_ple_norm.weight");
        L.layer_scalar = g("layer_scalar");
    }

    // Setup scheduler for GPU-accelerated encoder/LLM
    int n_be = 1;
    ggml_backend_t backends[2] = {ctx->backend, nullptr};
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
        backends[n_be++] = ctx->backend_cpu;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);

    // Allocate compute meta buffer for graph building (8 MB)
    ctx->compute_meta.resize(8 * 1024 * 1024);
    ctx->model_path = path_model;

    // Generate mel resources at runtime (Gemma4 GGUF doesn't include them)
    g4e_gen_hann_window(400, ctx->mel_window);
    g4e_gen_mel_filterbank(128, 400, 16000, ctx->mel_filterbank);

    // Look up special token IDs from vocab
    for (int i = 0; i < (int)m.vocab.size(); i++) {
        if (m.vocab[i] == "<bos>")
            ctx->bos_id = i;
        else if (m.vocab[i] == "<eos>" || m.vocab[i] == "</s>")
            ctx->eos_id = i;
        else if (m.vocab[i] == "<start_of_turn>")
            ctx->start_of_turn_id = i;
        else if (m.vocab[i] == "<end_of_turn>")
            ctx->end_of_turn_id = i;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "gemma4_e2b: audio %uL×%u, llm %uL×%u, vocab %u\n", ahp.num_layers, ahp.hidden_size,
                lhp.num_layers, lhp.hidden_size, lhp.vocab_size);
        fprintf(stderr, "gemma4_e2b: bos=%d eos=%d start_of_turn=%d end_of_turn=%d\n", ctx->bos_id, ctx->eos_id,
                ctx->start_of_turn_id, ctx->end_of_turn_id);
    }

    return ctx;
}

extern "C" char* gemma4_e2b_transcribe(struct gemma4_e2b_context* ctx, const float* pcm, int n_samples) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;

    auto& m = ctx->model;
    auto& ahp = m.audio_hp;
    auto& lhp = m.llm_hp;
    const bool verbose = ctx->verbosity >= 2 || getenv("GEMMA4_E2B_BENCH");
    const float eps = lhp.rms_norm_eps;

    if (ctx->verbosity >= 1)
        fprintf(stderr, "gemma4_e2b: %d samples (%.1fs)\n", n_samples, n_samples / 16000.0f);

    int64_t t0 = ggml_time_us();

    // ── Step 1: Mel spectrogram (128-bin, 16kHz, Whisper-style) ─────────
    const int n_fft = 400, hop = 160, n_mels = 128;
    const int n_freqs = n_fft / 2 + 1;

    // Prefer GGUF-stored mel resources; fall back to runtime-generated ones
    std::vector<float> hann_buf, filt_buf;
    const float* hann_ptr = ctx->mel_window.data();
    const float* filt_ptr = ctx->mel_filterbank.data();
    if (m.mel_window && m.mel_filters) {
        hann_buf.resize(n_fft);
        ggml_backend_tensor_get(m.mel_window, hann_buf.data(), 0, n_fft * sizeof(float));
        filt_buf.resize((size_t)n_freqs * n_mels);
        ggml_backend_tensor_get(m.mel_filters, filt_buf.data(), 0, filt_buf.size() * sizeof(float));
        hann_ptr = hann_buf.data();
        filt_ptr = filt_buf.data();
        if (verbose)
            fprintf(stderr, "gemma4_e2b: using GGUF-stored mel filterbank\n");
    }

    core_mel::Params mp;
    mp.n_fft = n_fft;
    mp.hop_length = hop;
    mp.win_length = n_fft;
    mp.n_mels = n_mels;
    mp.log_base = core_mel::LogBase::Log10;
    mp.log_guard = core_mel::LogGuard::MaxClip;
    mp.norm = core_mel::Normalization::GlobalClipMax;
    mp.layout = core_mel::Layout::MelsTime;
    mp.fb_layout = core_mel::FbLayout::FreqsMels;
    mp.matmul = core_mel::MatmulPrecision::Double;
    mp.log_eps = 1e-10f;
    mp.center_pad = true;
    mp.drop_last_frame = true;

    int T_mel = 0;
    auto mel = core_mel::compute(pcm, n_samples, hann_ptr, n_fft, filt_ptr, n_freqs, g4e_fft_wrapper, mp, T_mel);
    if (mel.empty()) {
        fprintf(stderr, "gemma4_e2b: mel computation failed\n");
        return nullptr;
    }
    // Cap to 30s
    if (T_mel > 3000)
        T_mel = 3000;

    if (verbose)
        fprintf(stderr, "gemma4_e2b: mel %dx%d (%.1f ms)\n", n_mels, T_mel, (ggml_time_us() - t0) / 1000.0);

    // ── Step 2-4: Encoder (Conv2D sub + Conformer + output proj) ────────
    int64_t t_enc0 = ggml_time_us();

    // Build single encoder graph: mel → conv2d sub → 12 conformer layers → output proj
    size_t enc_mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(32768, false);
    std::vector<uint8_t> enc_meta(enc_mem);
    ggml_init_params enc_ip = {enc_mem, enc_meta.data(), true};
    ggml_context* ectx = ggml_init(enc_ip);
    ggml_cgraph* enc_gf = ggml_new_graph_custom(ectx, 32768, false);

    // Input: mel [T_mel, n_mels, 1, 1] for conv2d
    ggml_tensor* mel_in = ggml_new_tensor_4d(ectx, GGML_TYPE_F32, T_mel, n_mels, 1, 1);
    ggml_set_name(mel_in, "mel");
    ggml_set_input(mel_in);

    // Conv2D subsampling layer 0: Conv2d(1→128, k=3, s=2, p=1) + RMSNorm + SiLU
    ggml_tensor* h = mel_in;
    if (m.sub_conv0_w) {
        h = ggml_conv_2d(ectx, m.sub_conv0_w, h, 2, 2, 1, 1, 1, 1);
        // h: [OW, OH, 128, 1] where OW≈T_mel/2, OH≈n_mels/2=64
        if (m.sub_norm0_w) {
            // RMSNorm over channel dim: reshape to [C, spatial]
            int ow = (int)h->ne[0], oh = (int)h->ne[1], c = (int)h->ne[2];
            h = ggml_reshape_2d(ectx, h, ow * oh, c);
            h = ggml_cont(ectx, ggml_transpose(ectx, h)); // [c, ow*oh]
            h = ggml_rms_norm(ectx, h, eps);
            h = ggml_mul(ectx, h, m.sub_norm0_w);
            h = ggml_cont(ectx, ggml_transpose(ectx, h)); // [ow*oh, c]
            h = ggml_reshape_4d(ectx, h, ow, oh, c, 1);
        }
        h = ggml_silu(ectx, h);
    }

    // Conv2D subsampling layer 1: Conv2d(128→32, k=3, s=2, p=1) + RMSNorm + SiLU
    if (m.sub_conv1_w) {
        h = ggml_conv_2d(ectx, m.sub_conv1_w, h, 2, 2, 1, 1, 1, 1);
        // h: [OW2, OH2, 32, 1] where OW2≈T_mel/4, OH2≈n_mels/4=32
        if (m.sub_norm1_w) {
            int ow = (int)h->ne[0], oh = (int)h->ne[1], c = (int)h->ne[2];
            h = ggml_reshape_2d(ectx, h, ow * oh, c);
            h = ggml_cont(ectx, ggml_transpose(ectx, h));
            h = ggml_rms_norm(ectx, h, eps);
            h = ggml_mul(ectx, h, m.sub_norm1_w);
            h = ggml_cont(ectx, ggml_transpose(ectx, h));
            h = ggml_reshape_4d(ectx, h, ow, oh, c, 1);
        }
        h = ggml_silu(ectx, h);
    }

    // Flatten: [OW2, OH2, 32, 1] → [32*OH2, OW2] = [1024, T_sub]
    int T_sub = (int)h->ne[0];
    int feat_dim = (int)h->ne[1] * (int)h->ne[2]; // OH2 * 32 = 32*32 = 1024
    // Need to reshape: permute to channel-first then flatten
    // h is [OW2, OH2, 32, 1]. We want [OH2*32, OW2] = [feat, T].
    // Reshape to [OW2, OH2*32, 1, 1] then transpose to [OH2*32, OW2]
    h = ggml_reshape_2d(ectx, h, T_sub, feat_dim);
    h = ggml_cont(ectx, ggml_transpose(ectx, h)); // [feat_dim, T_sub]

    // Input projection: Linear(1024→1024)
    if (m.sub_input_proj_w) {
        h = ggml_mul_mat(ectx, m.sub_input_proj_w, h); // [1024, T_sub]
    }

    int hidden = (int)ahp.hidden_size;

    // ── Conformer encoder (12 layers) ──
    for (uint32_t il = 0; il < ahp.num_layers; il++) {
        const auto& L = m.audio_layers[il];

        // Macaron FFN 1 (half-step)
        if (L.ffn1_up_w)
            h = build_macaron_ffn(ectx, h, L.ffn1_pre_ln, L.ffn1_up_w, L.ffn1_down_w, L.ffn1_post_ln,
                                  ahp.residual_weight, eps);

        // Self-attention (full, with per_dim_scale)
        h = build_conformer_self_attn(ectx, h, L, ahp);

        // LightConv1d
        if (L.conv_gate_w)
            h = build_light_conv1d(ectx, h, L, T_sub, hidden, eps);

        // Macaron FFN 2 (half-step)
        if (L.ffn2_up_w)
            h = build_macaron_ffn(ectx, h, L.ffn2_pre_ln, L.ffn2_up_w, L.ffn2_down_w, L.ffn2_post_ln,
                                  ahp.residual_weight, eps);

        // Output layer norm
        if (L.out_ln) {
            h = ggml_rms_norm(ectx, h, eps);
            h = ggml_mul(ectx, h, L.out_ln);
        }
    }

    // ── Output projection: Linear(1024→1536, bias) + embed proj ──
    if (m.audio_output_proj_w) {
        h = ggml_mul_mat(ectx, m.audio_output_proj_w, h);
        if (m.audio_output_proj_b)
            h = ggml_add(ectx, h, m.audio_output_proj_b);
    }
    if (m.audio_embed_proj_w)
        h = ggml_mul_mat(ectx, m.audio_embed_proj_w, h);

    ggml_set_name(h, "encoder_out");
    ggml_set_output(h);
    ggml_build_forward_expand(enc_gf, h);

    // Run encoder graph
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, enc_gf)) {
        fprintf(stderr, "gemma4_e2b: failed to alloc encoder graph\n");
        ggml_free(ectx);
        return nullptr;
    }

    // Set mel input data
    ggml_tensor* mel_t = ggml_graph_get_tensor(enc_gf, "mel");
    ggml_backend_tensor_set(mel_t, mel.data(), 0, (size_t)T_mel * n_mels * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, enc_gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gemma4_e2b: encoder graph compute failed\n");
        ggml_free(ectx);
        return nullptr;
    }

    // Read encoder output
    ggml_tensor* enc_out_t = ggml_graph_get_tensor(enc_gf, "encoder_out");
    int proj_dim = (int)enc_out_t->ne[0]; // should be 1536
    int N_audio = (int)enc_out_t->ne[1];  // T_sub
    std::vector<float> audio_emb((size_t)proj_dim * N_audio);
    ggml_backend_tensor_get(enc_out_t, audio_emb.data(), 0, audio_emb.size() * sizeof(float));
    ggml_free(ectx);

    if (verbose)
        fprintf(stderr, "gemma4_e2b: encoder done: %dx%d (%.1f ms)\n", proj_dim, N_audio,
                (ggml_time_us() - t_enc0) / 1000.0);

    // ── Step 5: Build prompt + inject audio ─────────────────────────────
    // Template: <bos><start_of_turn>user\n[audio_embeddings]Transcribe...<end_of_turn>\n<start_of_turn>model\n
    int64_t t_llm0 = ggml_time_us();
    const int d = (int)lhp.hidden_size;

    // Build prompt token sequence
    std::vector<int32_t> prompt_ids;
    prompt_ids.push_back(ctx->bos_id);
    if (ctx->start_of_turn_id >= 0)
        prompt_ids.push_back(ctx->start_of_turn_id);

    // Simple tokenization of "user\n" using vocab lookup
    // For now, use basic character-level fallback + known token IDs
    // TODO: proper BPE tokenization with core_bpe
    auto find_token = [&](const std::string& s) -> int {
        for (int i = 0; i < (int)m.vocab.size(); i++)
            if (m.vocab[i] == s)
                return i;
        return -1;
    };

    int user_id = find_token("user");
    int nl_id = find_token("\n");
    if (user_id >= 0)
        prompt_ids.push_back(user_id);
    if (nl_id >= 0)
        prompt_ids.push_back(nl_id);

    int audio_insert_pos = (int)prompt_ids.size(); // audio goes here

    // "Transcribe the following audio clip into text."
    // Simple approach: try to find common subword tokens
    std::vector<std::string> text_tokens = {"Transcribe", " the",  " following", " audio",
                                            " clip",      " into", " text",      "."};
    for (auto& w : text_tokens) {
        int tid = find_token(w);
        if (tid >= 0)
            prompt_ids.push_back(tid);
    }

    if (ctx->end_of_turn_id >= 0)
        prompt_ids.push_back(ctx->end_of_turn_id);
    if (nl_id >= 0)
        prompt_ids.push_back(nl_id);
    if (ctx->start_of_turn_id >= 0)
        prompt_ids.push_back(ctx->start_of_turn_id);
    int model_id = find_token("model");
    if (model_id >= 0)
        prompt_ids.push_back(model_id);
    if (nl_id >= 0)
        prompt_ids.push_back(nl_id);

    // Embed prompt tokens (with Gemma sqrt(d) scaling)
    float* prompt_emb = g4e_embed_tokens(ctx, prompt_ids.data(), (int)prompt_ids.size());
    if (!prompt_emb) {
        fprintf(stderr, "gemma4_e2b: failed to embed prompt tokens\n");
        return nullptr;
    }

    // Build combined embedding: [prefix_tokens | audio_embeddings | suffix_tokens]
    int n_prefix = audio_insert_pos;
    int n_suffix = (int)prompt_ids.size() - audio_insert_pos;
    int T_total = n_prefix + N_audio + n_suffix;

    std::vector<float> combined((size_t)d * T_total);

    // Copy prefix embeddings
    std::memcpy(combined.data(), prompt_emb, (size_t)d * n_prefix * sizeof(float));

    // Copy audio embeddings (proj_dim should == d == 1536)
    if (proj_dim == d) {
        std::memcpy(combined.data() + (size_t)d * n_prefix, audio_emb.data(), (size_t)d * N_audio * sizeof(float));
    } else {
        fprintf(stderr, "gemma4_e2b: proj_dim %d != llm_hidden %d — dimension mismatch\n", proj_dim, d);
        std::free(prompt_emb);
        return nullptr;
    }

    // Copy suffix embeddings
    std::memcpy(combined.data() + (size_t)d * (n_prefix + N_audio), prompt_emb + (size_t)d * n_prefix,
                (size_t)d * n_suffix * sizeof(float));
    std::free(prompt_emb);

    if (verbose)
        fprintf(stderr, "gemma4_e2b: prompt: %d prefix + %d audio + %d suffix = %d total\n", n_prefix, N_audio,
                n_suffix, T_total);

    // ── Step 6: Init KV cache and run prefill ───────────────────────────
    int max_ctx = std::max(4096, T_total + 512);
    if (!g4e_kv_init(ctx, max_ctx)) {
        fprintf(stderr, "gemma4_e2b: kv init failed\n");
        return nullptr;
    }

    // Prefill: run full prompt through LLM to fill KV cache
    float* prefill_logits = g4e_run_llm_kv(ctx, combined.data(), T_total, 0, nullptr, nullptr);
    if (!prefill_logits) {
        fprintf(stderr, "gemma4_e2b: prefill failed\n");
        return nullptr;
    }

    int vocab = (int)lhp.vocab_size;
    int first_token = core_greedy_decode::argmax(prefill_logits, vocab);
    std::free(prefill_logits);

    if (verbose)
        fprintf(stderr, "gemma4_e2b: prefill done, first_token=%d (%.1f ms)\n", first_token,
                (ggml_time_us() - t_llm0) / 1000.0);

    // ── Step 7: Greedy decode ───────────────────────────────────────────
    core_greedy_decode::Config cfg;
    cfg.max_new_tokens = 256;
    cfg.eos_id = ctx->eos_id;
    cfg.vocab_size = vocab;
    cfg.temperature = ctx->temperature;

    auto gen = core_greedy_decode::run(ctx, first_token, T_total, g4e_embed_tokens, g4e_run_llm_kv, cfg);

    if (verbose)
        fprintf(stderr, "gemma4_e2b: decoded %d tokens (%.1f ms total)\n", (int)gen.size(),
                (ggml_time_us() - t0) / 1000.0);

    // ── Step 8: Detokenize ──────────────────────────────────────────────
    std::string result;
    for (int tid : gen) {
        if (tid == ctx->bos_id || tid == ctx->eos_id)
            continue;
        if (tid == ctx->start_of_turn_id || tid == ctx->end_of_turn_id)
            continue;
        if (tid >= 0 && tid < (int)m.vocab.size())
            result += m.vocab[tid];
    }

    // Strip leading/trailing whitespace
    size_t s = result.find_first_not_of(" \n\t\r");
    size_t e = result.find_last_not_of(" \n\t\r");
    if (s != std::string::npos && e != std::string::npos)
        result = result.substr(s, e - s + 1);

    return strdup(result.c_str());
}

extern "C" void gemma4_e2b_free(struct gemma4_e2b_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf_w)
        ggml_backend_buffer_free(ctx->model.buf_w);
    if (ctx->model.ctx_w)
        ggml_free(ctx->model.ctx_w);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" void gemma4_e2b_set_n_threads(struct gemma4_e2b_context* ctx, int n_threads) {
    if (ctx && n_threads > 0) {
        ctx->n_threads = n_threads;
        if (ctx->backend_cpu)
            ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    }
}
