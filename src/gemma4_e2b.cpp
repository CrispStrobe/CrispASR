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
    uint32_t head_dim = 256;        // sliding-attention layers
    uint32_t global_head_dim = 512; // full-attention layers
    uint32_t intermediate_size = 6144;
    uint32_t vocab_size = 262144;
    uint32_t max_position_embeddings = 131072;
    uint32_t sliding_window = 512;
    uint32_t num_kv_shared_layers = 0;  // first N layers reuse later K/V
    float rope_theta = 10000.0f;        // sliding layers
    float rope_theta_full = 1000000.0f; // full layers (partial-rotary)
    float partial_rotary_factor = 0.25f;
    float final_logit_softcapping = 30.0f;
    float rms_norm_eps = 1e-6f;
    bool use_double_wide_mlp = false;
    bool attention_k_eq_v = false;
    // 1 = full attention, 0 = sliding. Indexed by layer; size = num_layers.
    std::vector<int32_t> layer_full_mask;
    // KV-cache sharing (Gemma4-style). The LAST `num_kv_shared_layers`
    // layers don't compute their own K/V — they reuse K/V from an earlier
    // donor layer of the same `layer_type`. -1 = compute own K/V; non-
    // negative = read K/V from cache slot at this donor index. Built once
    // after the per-layer mask is known (see g4e_compute_kv_share_donor).
    std::vector<int32_t> kv_share_donor;
};

// ── Model tensors ───────────────────────────────────────────────────────────

// Per-Linear QAT clipping bounds. Gemma4ClippableLinear applies
//   x = clamp(x, in_min, in_max); y = W @ x; y = clamp(y, out_min, out_max).
// At init we read the four 1-element scalars from the GGUF (or default
// to ±inf when absent on older GGUFs that filtered them).
struct g4e_clip {
    float in_min = -INFINITY;
    float in_max = INFINITY;
    float out_min = -INFINITY;
    float out_max = INFINITY;
};

struct g4e_audio_layer {
    // Macaron FFN 1
    ggml_tensor* ffn1_pre_ln = nullptr;
    ggml_tensor* ffn1_up_w = nullptr;
    ggml_tensor* ffn1_down_w = nullptr;
    ggml_tensor* ffn1_post_ln = nullptr;
    g4e_clip clip_ffn1_up, clip_ffn1_down;

    // Self-attention
    ggml_tensor* attn_pre_ln = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_per_dim_scale = nullptr; // [head_dim]
    ggml_tensor* attn_rel_k_w = nullptr;       // [hidden, hidden] — relative position bias
    ggml_tensor* attn_post_ln = nullptr;
    g4e_clip clip_q, clip_k, clip_v, clip_o;

    // LightConv1d
    ggml_tensor* conv_pre_ln = nullptr;
    ggml_tensor* conv_gate_w = nullptr; // [2*hidden, hidden] — GLU gating
    ggml_tensor* conv_dw_w = nullptr;   // [hidden, 1, k] — depthwise conv
    ggml_tensor* conv_ln = nullptr;
    ggml_tensor* conv_out_w = nullptr; // [hidden, hidden]
    g4e_clip clip_conv_gate, clip_conv_out;

    // Macaron FFN 2
    ggml_tensor* ffn2_pre_ln = nullptr;
    ggml_tensor* ffn2_up_w = nullptr;
    ggml_tensor* ffn2_down_w = nullptr;
    g4e_clip clip_ffn2_up, clip_ffn2_down;
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
    ggml_tensor* llm_ple_w = nullptr;      // [vocab_per_layer, n_layers*ple_dim] — per-layer embeddings
    ggml_tensor* llm_final_norm = nullptr; // [hidden]
    // Per-layer-input projection from inputs_embeds (Gemma4):
    //   per_layer_proj  = Linear(hidden -> n_layers*ple_dim)  bias=False
    //   per_layer_norm  = RMSNorm(ple_dim)
    // The mixed input fed into each layer's PLE block is:
    //   pli = (norm(per_layer_proj(inputs_embeds) * 1/sqrt(hidden))
    //        + embed_tokens_per_layer(input_ids) * sqrt(ple_dim)) * 1/sqrt(2)
    ggml_tensor* llm_per_layer_proj_w = nullptr;    // [hidden, n_layers*ple_dim]
    ggml_tensor* llm_per_layer_proj_norm = nullptr; // [ple_dim]

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

    // KV cache for LLM decode.
    //
    // Gemma4 alternates sliding-window (head_dim=256) and full-attention
    // (head_dim=global_head_dim, e.g. 512) layers, so they need separate
    // KV cache buffers — one ggml tensor can't hold rows of two different
    // widths along its inner axis. `kv_k` / `kv_v` are the LOCAL (sliding)
    // cache; `kv_k_full` / `kv_v_full` are the GLOBAL (full-attention)
    // cache. Both store all `num_layers` slots so layer-index lookups
    // stay simple — only the slots whose `layer_full_mask` matches that
    // cache are written/read.
    ggml_context* kv_ctx = nullptr;
    ggml_tensor* kv_k = nullptr; // local (sliding) layers
    ggml_tensor* kv_v = nullptr;
    ggml_tensor* kv_k_full = nullptr; // full-attention layers
    ggml_tensor* kv_v_full = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    int kv_max_ctx = 0;

    // Pre-computed mel resources (generated at init, not stored in GGUF)
    std::vector<float> mel_window;     // Hann window [n_fft]
    std::vector<float> mel_filterbank; // [n_freqs * n_mels]

    // Special token IDs (looked up at init from vocab)
    int bos_id = 2;
    int eos_id = 1;
    int pad_id = 0;
    int start_of_turn_id = -1;
    int end_of_turn_id = -1;
    // Audio soft-token id, used to wrap the audio span in the prompt so
    // the model can find/locate the audio modality. The PLE lookup at
    // audio positions, however, should NOT use this id — see the comment
    // in gemma4_e2b_transcribe and HF Gemma4Model.forward L2215, which
    // replaces multimodal token ids with pad_token_id before embedding.
    int audio_soft_token_id = -1;

    // Per-token IDs for the next g4e_run_llm_kv call. Populated by
    // gemma4_e2b_transcribe (prefill — uses audio_soft_token_id for the
    // audio span) and by g4e_embed_tokens (decode — single token). The
    // graph reads this via the `ple_ids` input tensor.
    std::vector<int32_t> ple_token_ids;
};

// ── Conformer encoder graph builder ─────────────────────────────────────────
// Builds a single ggml graph for the full 12-layer USM Conformer encoder.
// Input: mel features [n_mels, T_mel] after Conv2D subsampling → [hidden, T_sub]
// Output: [output_proj_dims, T_sub]

// Apply Gemma4ClippableLinear semantics: x = clamp(x); y = W @ x; y = clamp(y).
//
// Important: ggml_clamp is in-place — returns a view of the input and
// overwrites its data when the graph runs. The caller is responsible
// for ensuring `x` is a private buffer (not shared with sibling
// consumers). For Q/K/V projections that share the same input, the
// caller should pre-cont the shared input ONCE and pass the same
// per-projection-private clone in. The output clamp is always safe in
// place because mul_mat produces a fresh tensor.
//
// `private_input == false` triggers an internal ggml_cont() — convenient
// for one-off uses where the caller doesn't want to manage the copy
// (e.g. ffn1.up only uses h once).
static inline ggml_tensor* g4e_clipped_mul_mat(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, const g4e_clip& c,
                                               bool private_input = false) {
    if (std::isfinite(c.in_min) || std::isfinite(c.in_max)) {
        if (!private_input)
            x = ggml_cont(ctx, x); // private copy
        x = ggml_clamp(ctx, x, c.in_min, c.in_max);
    }
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    if (std::isfinite(c.out_min) || std::isfinite(c.out_max))
        y = ggml_clamp(ctx, y, c.out_min, c.out_max);
    return y;
}

static ggml_tensor* build_macaron_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pre_ln, ggml_tensor* up_w,
                                      ggml_tensor* down_w, ggml_tensor* post_ln, float residual_weight, float eps,
                                      const g4e_clip& clip_up, const g4e_clip& clip_down) {
    // Half-step FFN with QAT clipping on both projections.
    // x + residual_weight * post_ln(clipped_down(silu(clipped_up(pre_ln(x)))))
    ggml_tensor* h = ggml_rms_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, pre_ln);
    h = g4e_clipped_mul_mat(ctx, up_w, h, clip_up);
    h = ggml_silu(ctx, h);
    h = g4e_clipped_mul_mat(ctx, down_w, h, clip_down);
    ggml_tensor* normed = ggml_rms_norm(ctx, h, eps);
    normed = ggml_mul(ctx, normed, post_ln);
    return ggml_add(ctx, x, ggml_scale(ctx, normed, residual_weight));
}

static ggml_tensor* build_light_conv1d(ggml_context* ctx, ggml_tensor* x, const g4e_audio_layer& L, int T, int hidden,
                                       float eps) {
    // LightConv1d with QAT clipping on linear_start (gate) and linear_end (out).
    ggml_tensor* residual = x;
    ggml_tensor* h = ggml_rms_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, L.conv_pre_ln);

    // GLU gating: gate_proj produces [2*hidden, T], split into value + gate
    h = g4e_clipped_mul_mat(ctx, L.conv_gate_w, h, L.clip_conv_gate); // [2*hidden, T]
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

    // Conv norm + activation + clipped out projection.
    h = ggml_rms_norm(ctx, h, eps);
    h = ggml_mul(ctx, h, L.conv_ln);
    h = ggml_silu(ctx, h);
    h = g4e_clipped_mul_mat(ctx, L.conv_out_w, h, L.clip_conv_out);

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

    // HTK triangular filters. Gemma4AudioFeatureExtractor uses
    // norm=None (no Slaney area-normalization). We had Slaney enabled
    // before — that's wrong for Gemma4.
    for (int m = 0; m < n_mels; m++) {
        double lo = mel_pts[m], ctr = mel_pts[m + 1], hi = mel_pts[m + 2];
        for (int f = 0; f < n_freqs; f++) {
            double val = 0.0;
            if (fft_freqs[f] >= lo && fft_freqs[f] <= ctr && ctr > lo)
                val = (fft_freqs[f] - lo) / (ctr - lo);
            else if (fft_freqs[f] > ctr && fft_freqs[f] <= hi && hi > ctr)
                val = (hi - fft_freqs[f]) / (hi - ctr);
            // fb layout: (n_freqs, n_mels) for FbLayout::FreqsMels
            fb[(size_t)f * n_mels + m] = (float)val;
        }
    }
}

// HF-faithful Gemma4 mel computation. Bypasses core_mel because the
// HF FE uses a unique combination of:
//   - semicausal padding (frame_length//2 zeros at start, no end pad)
//   - unfold with size=frame_length+1 (then drop last sample → 320)
//   - magnitude (|stft|) projection through HTK no-norm filterbank
//   - log(mel + mel_floor) (additive epsilon, natural log)
//
// Output layout: (T_mel, n_mels) row-major — data[t*n_mels + m]. This
// matches HF's natural FE output and binds 1:1 to a ggml tensor of
// ne=(n_mels, T_mel, 1, 1) (n_mels=fast), which is the conv2d input
// shape that mirrors HF's (B, 1, T, n_mels) interpretation (T as H,
// n_mels as W). The crispasr-diff hook passes this layout through.
static std::vector<float> g4e_compute_mel_hf_faithful(const float* pcm, int n_samples, int n_fft, int win_length,
                                                      int hop_length, int n_mels,
                                                      const float* window /* size win_length */,
                                                      const float* mel_filters /* (n_freqs, n_mels) */, float mel_floor,
                                                      int& out_T_mel) {
    const int n_freqs = n_fft / 2 + 1;
    const int pad_left = win_length / 2;
    const int total = n_samples + pad_left;
    const int frame_size = win_length + 1; // unfold size; drop last sample later
    if (total < frame_size) {
        out_T_mel = 0;
        return {};
    }
    const int T = (total - frame_size) / hop_length + 1;

    std::vector<float> mel_out((size_t)T * n_mels, 0.0f);
    std::vector<float> frame(n_fft, 0.0f);
    std::vector<float> fft_buf((size_t)n_fft * 2, 0.0f);
    std::vector<float> magnitude((size_t)n_freqs);

    for (int t = 0; t < T; t++) {
        std::fill(frame.begin(), frame.end(), 0.0f);
        for (int s = 0; s < win_length; s++) {
            const int abs_idx = t * hop_length + s;
            const int pcm_idx = abs_idx - pad_left;
            const float v = (pcm_idx >= 0 && pcm_idx < n_samples) ? pcm[pcm_idx] : 0.0f;
            frame[s] = v * window[s];
        }
        g4e_fft_wrapper(frame.data(), n_fft, fft_buf.data());

        for (int k = 0; k < n_freqs; k++) {
            const float re = fft_buf[2 * k];
            const float im = fft_buf[2 * k + 1];
            magnitude[k] = std::sqrt(re * re + im * im);
        }

        // Mel projection + log directly into (T, n_mels) layout.
        float* row = mel_out.data() + (size_t)t * n_mels;
        for (int m = 0; m < n_mels; m++)
            row[m] = 0.0f;
        for (int k = 0; k < n_freqs; k++) {
            const float* fb_row = mel_filters + (size_t)k * n_mels;
            const float mag = magnitude[k];
            for (int m = 0; m < n_mels; m++)
                row[m] += mag * fb_row[m];
        }
        for (int m = 0; m < n_mels; m++)
            row[m] = std::log(row[m] + mel_floor);
    }
    out_T_mel = T;
    return mel_out;
}

static void g4e_gen_hann_window(int n_fft, std::vector<float>& win) {
    win.resize(n_fft);
    for (int i = 0; i < n_fft; i++)
        win[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / n_fft));
}

// Bake `q_scale * softplus(per_dim_scale)` into the audio attention's
// per_dim_scale tensor, so the inference-time graph just multiplies Q by
// it (matching HF Gemma4AudioAttention.forward L278).
//
// HF formula:
//     q_scale = head_dim^-0.5 / log(2)
//     softplus(x) = log(1 + exp(x))
//     query_states *= q_scale * softplus(per_dim_scale)
//
// per_dim_scale is initialised to zeros and trained, so softplus(0) = log(2)
// gives an effective initial scale of 1/sqrt(head_dim) — i.e. plain
// 1/sqrt(d) attention. After training the per-dim values shift away from
// zero. We do this once at load time because the input to softplus is a
// (small, ~head_dim-sized) constant tensor that doesn't depend on the
// activations, so there's no value in recomputing it every forward pass.
static void g4e_bake_audio_per_dim_scale(g4e_model& m) {
    const int hd = (int)(m.audio_hp.hidden_size / m.audio_hp.num_heads);
    if (hd <= 0)
        return;
    const float q_scale = (1.0f / std::sqrt((float)hd)) / std::log(2.0f);
    std::vector<float> buf((size_t)hd);
    for (auto& L : m.audio_layers) {
        if (!L.attn_per_dim_scale)
            continue;
        if ((int)L.attn_per_dim_scale->ne[0] != hd)
            continue;
        ggml_backend_tensor_get(L.attn_per_dim_scale, buf.data(), 0, (size_t)hd * sizeof(float));
        for (int i = 0; i < hd; i++) {
            const float x = buf[i];
            // Numerically stable softplus: for x >> 0, log1p(exp(x)) ≈ x.
            // For x << 0, log1p(exp(x)) ≈ exp(x) and is small. The general
            // form below uses log1p + abs to avoid overflow at large x.
            const float sp = (x > 20.0f) ? x : (x < -20.0f ? std::exp(x) : std::log1p(std::exp(x)));
            buf[i] = q_scale * sp;
        }
        ggml_backend_tensor_set(L.attn_per_dim_scale, buf.data(), 0, (size_t)hd * sizeof(float));
    }
}

// ── KV-cache sharing (Gemma4) ──────────────────────────────────────────────
// Mirrors transformers/models/gemma4/modeling_gemma4.py:
//
//   first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers
//   is_kv_shared_layer        = layer_idx >= first_kv_shared_layer_idx > 0
//   kv_shared_layer_index     = last layer in [0, first_kv_shared_layer_idx)
//                               with the SAME layer_type as this layer.
//
// The shared layer does NOT compute its own K/V (and HF doesn't even
// allocate k_proj/v_proj/k_norm/v_norm for it); it reads the donor's
// already-RoPE'd, already-cached K/V instead. Same `layer_type`
// guarantees same head_dim, same RoPE, same kv buffer.
static void g4e_compute_kv_share_donor(g4e_llm_hp& lhp) {
    const int N = (int)lhp.num_layers;
    const int Ns = (int)lhp.num_kv_shared_layers;
    lhp.kv_share_donor.assign(N, -1);
    if (Ns <= 0 || Ns >= N || (int)lhp.layer_full_mask.size() < N)
        return;
    const int first_shared = N - Ns;
    for (int il = first_shared; il < N; il++) {
        const int my_type = lhp.layer_full_mask[il] ? 1 : 0;
        for (int j = first_shared - 1; j >= 0; j--) {
            const int j_type = lhp.layer_full_mask[j] ? 1 : 0;
            if (j_type == my_type) {
                lhp.kv_share_donor[il] = j;
                break;
            }
        }
    }
}

// ── KV cache init ──────────────────────────────────────────────────────────

static bool g4e_kv_init(gemma4_e2b_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    if (ctx->kv_k)
        return true;

    const auto& lhp = ctx->model.llm_hp;
    const int hd = (int)lhp.head_dim;
    const int hd_full = (int)lhp.global_head_dim;
    const int n_kv = (int)lhp.num_kv_heads;
    const int n_lay = (int)lhp.num_layers;

    // Are there any full-attention layers in this model? If layer_full_mask
    // is all-zeros (older GGUF, or a model without the alternation) we skip
    // the second buffer entirely.
    bool has_full = false;
    for (int v : lhp.layer_full_mask)
        if (v) {
            has_full = true;
            break;
        }

    ggml_init_params kp = {ggml_tensor_overhead() * 8 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");
    if (has_full && hd_full != hd) {
        ctx->kv_k_full = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd_full, max_ctx, n_kv, n_lay);
        ctx->kv_v_full = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd_full, max_ctx, n_kv, n_lay);
        ggml_set_name(ctx->kv_k_full, "kv_k_full");
        ggml_set_name(ctx->kv_v_full, "kv_v_full");
    }

    size_t kbytes = ggml_nbytes(ctx->kv_k);
    size_t vbytes = ggml_nbytes(ctx->kv_v);
    size_t kfbytes = ctx->kv_k_full ? ggml_nbytes(ctx->kv_k_full) : 0;
    size_t vfbytes = ctx->kv_v_full ? ggml_nbytes(ctx->kv_v_full) : 0;
    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, kbytes + vbytes + kfbytes + vfbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "gemma4_e2b: failed to alloc kv buffer\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    if (ctx->kv_k_full)
        ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k_full, base + kbytes + vbytes);
    if (ctx->kv_v_full)
        ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v_full, base + kbytes + vbytes + kfbytes);
    ctx->kv_max_ctx = max_ctx;

    if (ctx->verbosity >= 1)
        fprintf(stderr, "gemma4_e2b: kv cache %d MiB (hd=%d max_ctx=%d n_kv=%d n_lay=%d)\n",
                (int)((kbytes + vbytes) / 1048576), hd, max_ctx, n_kv, n_lay);
    return true;
}

// Build a chunked-local attention mask matching HF Gemma4AudioAttention.
// Each block of `chunk_size` Q positions attends to a K window covering
// the block plus `context_left - 1` past frames and `context_right`
// future frames. Positions outside that window are masked with -inf.
//
// Returned vector has shape [T, T] in row-major (q outer, k inner) — the
// same layout ggml_flash_attn_ext expects when given mask shape (Lk, Lq)
// after a contiguous F16 tensor view: actually flash_attn_ext expects
// mask shape (n_kv_heads, n_padded, T) so we just hand it the smallest
// equivalent buffer via a 2D F16 tensor of shape [n_kv_padded, T].
static std::vector<ggml_fp16_t> g4e_build_audio_attn_mask(int T, int chunk_size, int context_left, int context_right) {
    const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask((size_t)T * (size_t)T, neginf_h);
    const int max_past = context_left - 1; // HF: max_past_horizon
    for (int q = 0; q < T; q++) {
        const int b = q / chunk_size;
        const int blk_start = b * chunk_size;
        const int k_lo = std::max(0, blk_start - max_past);
        const int k_hi = std::min(T, blk_start + chunk_size + context_right);
        for (int k = k_lo; k < k_hi; k++)
            mask[(size_t)q * T + k] = zero_h;
    }
    return mask;
}

// HF Gemma4AudioRelPositionalEncoding-style sinusoidal pos enc.
// Produces (hidden_size, n_pos=13) row-major where n_pos is the number
// of relative-position bias positions (== chunk_size + 1 in HF). The
// position ids are arange(12, -1, -1) = [12, 11, ..., 1, 0].
static std::vector<float> g4e_make_audio_pos_enc(int hidden_size, int chunk_size) {
    const int n_pos = chunk_size + 1; // 13 for chunk_size=12
    const int half = hidden_size / 2;
    std::vector<float> pe((size_t)hidden_size * n_pos, 0.0f);
    const float min_t = 1.0f, max_t = 10000.0f;
    const float log_inc = std::log(max_t / min_t) / std::max(half - 1, 1);
    for (int p = 0; p < n_pos; p++) {
        const float pos_id = (float)(chunk_size - p); // arange(12, -1, -1)[p]
        for (int i = 0; i < half; i++) {
            const float inv_t = min_t * std::exp(-i * log_inc);
            const float scaled = pos_id * inv_t;
            // HF concat layout: [sin..., cos...] with n_timescales=hidden/2 each.
            pe[(size_t)p * hidden_size + i] = std::sin(scaled);
            pe[(size_t)p * hidden_size + half + i] = std::cos(scaled);
        }
    }
    return pe;
}

// ── Conformer self-attention with chunked-local + relative position bias ──
// HF-faithful implementation of Gemma4AudioAttention.forward. Block-wise
// manual attention because the rel_pos_bias is a per-block computation
// that doesn't fit ggml's flash_attn_ext mask interface (the mask is
// computed from Q via the relative_k_proj, not a static tensor).
//
// pos_enc: ne=(hidden_size, n_pos=13). Same for every layer.
// pad_mask: ne=(context_size, num_blocks). Per-block boundary mask
//   marking K positions outside [0, T) as -inf, others 0.

static ggml_tensor* build_conformer_self_attn(ggml_context* ctx, ggml_tensor* x, const g4e_audio_layer& L,
                                              const g4e_audio_hp& hp, ggml_tensor* pos_enc, ggml_tensor* pad_mask) {
    const int hd = (int)hp.head_dim;
    const int n_h = (int)hp.num_heads;
    const int hidden = (int)hp.hidden_size;
    const int T = (int)x->ne[1];
    const int chunk_size = (int)hp.chunk_size;
    const int max_past = (int)hp.context_left - 1;
    const int max_future = 0;                                    // E2B: attention_context_right=0
    const int context_size = chunk_size + max_past + max_future; // 24
    const int n_pos = chunk_size + 1;                            // 13
    const int num_blocks = (T + chunk_size - 1) / chunk_size;
    const int T_padded = num_blocks * chunk_size;
    const float eps = 1e-6f;
    const float softcap = hp.attention_logit_cap;

    ggml_tensor* residual = x;
    ggml_tensor* h = ggml_rms_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, L.attn_pre_ln);

    // Q/K/V projections + scaling. The Q/K/V proj inputs share `h` and
    // the in-place clamp would corrupt sibling reads. Pre-cont once into
    // 3 private clones (3 ops) instead of letting clipped_mul_mat cont
    // 3 times (saves nothing here, BUT we now skip the redundant cont
    // when input is already a private buffer).
    ggml_tensor* h_q = ggml_cont(ctx, h);
    ggml_tensor* h_k = ggml_cont(ctx, h);
    ggml_tensor* h_v = ggml_cont(ctx, h);
    ggml_tensor* Q = g4e_clipped_mul_mat(ctx, L.attn_q_w, h_q, L.clip_q, /*private_input=*/true);
    ggml_tensor* K = g4e_clipped_mul_mat(ctx, L.attn_k_w, h_k, L.clip_k, /*private_input=*/true);
    ggml_tensor* V = g4e_clipped_mul_mat(ctx, L.attn_v_w, h_v, L.clip_v, /*private_input=*/true);

    Q = ggml_reshape_3d(ctx, Q, hd, n_h, T);
    K = ggml_reshape_3d(ctx, K, hd, n_h, T);
    V = ggml_reshape_3d(ctx, V, hd, n_h, T);

    if (L.attn_per_dim_scale) {
        ggml_tensor* scale = ggml_reshape_3d(ctx, L.attn_per_dim_scale, hd, 1, 1);
        Q = ggml_mul(ctx, Q, scale);
    }
    {
        const float k_scale = std::log1p(std::exp(1.0f)) / std::log(2.0f);
        K = ggml_scale(ctx, K, k_scale);
    }

    // Permute (D, H, T) → (D, T, H) for time-axis chunking.
    // ggml_permute(a, p0, p1, p2, p3): pN = output axis for input axis N.
    // input 0 (D)→0; input 1 (H)→2; input 2 (T)→1; input 3→3.
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // (D, T, H)
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Pad along T axis: Q gets end-padding to T_padded; K/V get max_past
    // at start and (T_padded - T) + max_future + chunk_size - 1 at end.
    // ggml_pad_ext args: (ctx, x, l0, r0, l1, r1, l2, r2, l3, r3).
    if (T_padded > T) {
        Q = ggml_pad_ext(ctx, Q, 0, 0, 0, T_padded - T, 0, 0, 0, 0);
    }
    const int K_pad_start = max_past;
    const int K_pad_end = (T_padded - T) + max_future + chunk_size - 1;
    K = ggml_pad_ext(ctx, K, 0, 0, K_pad_start, K_pad_end, 0, 0, 0, 0);
    V = ggml_pad_ext(ctx, V, 0, 0, K_pad_start, K_pad_end, 0, 0, 0, 0);

    // Compute relative_k_states: rel_K_proj(pos_enc) → (hd*n_h, n_pos).
    // pos_enc ne=(hidden, n_pos). rel_k weight is HF (hidden, n_h*hd) →
    // ggml ne=(hidden, n_h*hd). mul_mat gives (n_h*hd, n_pos).
    ggml_tensor* rel_K = ggml_mul_mat(ctx, L.attn_rel_k_w, pos_enc); // (hd*n_h, n_pos)
    rel_K = ggml_reshape_3d(ctx, rel_K, hd, n_h, n_pos);             // (D, H, n_pos)
    // Permute to (D, n_pos, H) for matmul with Q_block.
    // input 0 (D)→0; input 1 (H)→2; input 2 (n_pos)→1.
    rel_K = ggml_cont(ctx, ggml_permute(ctx, rel_K, 0, 2, 1, 3)); // (D, n_pos, H)

    // Per-block manual attention.
    std::vector<ggml_tensor*> block_outs;
    block_outs.reserve(num_blocks);
    for (int b = 0; b < num_blocks; b++) {
        // Q_block view: (D, chunk_size, H) at q_offset = b*chunk_size frames.
        const size_t q_off = (size_t)b * chunk_size * Q->nb[1];
        ggml_tensor* Qb = ggml_view_3d(ctx, Q, hd, chunk_size, n_h, Q->nb[1], Q->nb[2], q_off);
        Qb = ggml_cont(ctx, Qb);

        // K_window view: (D, context_size, H) at k_offset = b*chunk_size frames in padded K.
        const size_t k_off = (size_t)b * chunk_size * K->nb[1];
        ggml_tensor* Kw = ggml_view_3d(ctx, K, hd, context_size, n_h, K->nb[1], K->nb[2], k_off);
        Kw = ggml_cont(ctx, Kw);

        ggml_tensor* Vw = ggml_view_3d(ctx, V, hd, context_size, n_h, V->nb[1], V->nb[2], k_off);
        Vw = ggml_cont(ctx, Vw);

        // matrix_ac = K_window^T @ Q_block via mul_mat(Kw, Qb).
        // ggml_mul_mat(A, B) where A and B share ne[0] as contracted dim:
        //   A=(D, ctx, H), B=(D, chunk, H) → C=(ctx, chunk, H).
        ggml_tensor* matrix_ac = ggml_mul_mat(ctx, Kw, Qb); // (ctx_size, chunk, H)

        // matrix_bd_raw = rel_K^T @ Q_block: rel_K=(D, n_pos, H), Q=(D, chunk, H)
        //   → (n_pos, chunk, H).
        ggml_tensor* bd_raw = ggml_mul_mat(ctx, rel_K, Qb); // (n_pos, chunk, H)

        // rel_shift: input (n_pos=13, chunk_size, H) → output (context_size=24, chunk_size, H).
        // PyTorch shift on (chunk, n_pos): pad last dim to (chunk, ctx_size+1=25), flatten,
        // trim to chunk*ctx_size, reshape (chunk, ctx_size).
        // In ggml with (n_pos, chunk, H): pad ne[0] from n_pos to ctx_size+1=25:
        ggml_tensor* bd_pad = ggml_pad_ext(ctx, bd_raw, 0, (context_size + 1) - n_pos, 0, 0, 0, 0, 0, 0);
        // ne=(ctx_size+1, chunk, H). Total elem per head: (ctx_size+1)*chunk = 25*12 = 300.
        // Reshape to flat (ctx_size+1)*chunk per head, take first ctx_size*chunk = 24*12 = 288.
        // Then reshape to (ctx_size, chunk, H).
        ggml_tensor* bd_flat = ggml_reshape_2d(ctx, bd_pad, (context_size + 1) * chunk_size, n_h);
        ggml_tensor* bd_trim = ggml_view_2d(ctx, bd_flat, context_size * chunk_size, n_h, bd_flat->nb[1], 0);
        bd_trim = ggml_cont(ctx, bd_trim);
        ggml_tensor* matrix_bd = ggml_reshape_3d(ctx, bd_trim, context_size, chunk_size, n_h);

        // Sum + softcap + boundary mask.
        // DEBUG: toggle to disable rel_pos_bias contribution.
        ggml_tensor* scores = std::getenv("CRISPASR_NO_REL_POS") ? matrix_ac : ggml_add(ctx, matrix_ac, matrix_bd);
        (void)matrix_bd;
        scores = ggml_scale(ctx, scores, 1.0f / softcap);
        scores = ggml_tanh(ctx, scores);
        scores = ggml_scale(ctx, scores, softcap);

        // Boundary mask for this block: (context_size, 1, 1) view of pad_mask
        // at column b. ggml_add_inplace can broadcast across (chunk, H).
        ggml_tensor* mask_b = ggml_view_1d(ctx, pad_mask, context_size, (size_t)b * pad_mask->nb[1]);
        // Convert F16 to F32 for adding to scores. Use ggml_cast or ggml_cpy.
        // Simpler: cast via ggml_cpy to a new f32 tensor of same shape.
        ggml_tensor* mask_b_f32 = ggml_cast(ctx, mask_b, GGML_TYPE_F32);
        // Broadcast add: (ctx, 1, 1) + (ctx, chunk, H) → (ctx, chunk, H).
        ggml_tensor* mask_b_3d = ggml_reshape_3d(ctx, mask_b_f32, context_size, 1, 1);
        scores = ggml_add(ctx, scores, mask_b_3d);

        // Softmax along ne[0] = context_size (the K dim).
        scores = ggml_soft_max(ctx, scores);

        // attn_block = V_window @ scores along ctx_size.
        // V_window=(D, ctx, H). To contract along ctx, permute V to (ctx, D, H).
        // ggml_permute: input 0 (D)→1; input 1 (ctx)→0; input 2 (H)→2.
        ggml_tensor* Vw_p = ggml_cont(ctx, ggml_permute(ctx, Vw, 1, 0, 2, 3)); // (ctx, D, H)
        // mul_mat(Vw_p, scores): Vw_p=(ctx, D, H), scores=(ctx, chunk, H) → (D, chunk, H).
        ggml_tensor* attn_block = ggml_mul_mat(ctx, Vw_p, scores); // (D, chunk, H)
        block_outs.push_back(attn_block);
    }

    // Concat blocks along T (ne[1]).
    ggml_tensor* attn_out = block_outs[0];
    for (int b = 1; b < num_blocks; b++)
        attn_out = ggml_concat(ctx, attn_out, block_outs[b], 1);
    // attn_out ne=(D, T_padded, H). Trim to T frames.
    if (T_padded > T) {
        attn_out = ggml_view_3d(ctx, attn_out, hd, T, n_h, attn_out->nb[1], attn_out->nb[2], 0);
        attn_out = ggml_cont(ctx, attn_out);
    }
    // Permute back (D, T, H) → (D, H, T): input 0→0; input 1 (T)→2; input 2 (H)→1.
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3)); // (D, H, T)
    attn_out = ggml_reshape_2d(ctx, attn_out, hd * n_h, T);

    // Output projection.
    attn_out = g4e_clipped_mul_mat(ctx, L.attn_o_w, attn_out, L.clip_o);

    // Post-norm + residual.
    attn_out = ggml_rms_norm(ctx, attn_out, eps);
    attn_out = ggml_mul(ctx, attn_out, L.attn_post_ln);
    (void)hidden;
    return ggml_add(ctx, residual, attn_out);
}

// Old signature for compatibility — DEAD (replaced by the rel_pos version).
static ggml_tensor* build_conformer_self_attn_legacy(ggml_context* ctx, ggml_tensor* x, const g4e_audio_layer& L,
                                                     const g4e_audio_hp& hp, ggml_tensor* attn_mask) {
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

    // Q per-dim scale. HF Gemma4AudioAttention.forward L278:
    //   query_states = query_states * q_scale * softplus(per_dim_scale)
    //   q_scale      = head_dim^-0.5 / log(2)
    // We bake `q_scale * softplus(per_dim_scale)` into attn_per_dim_scale
    // on the host once at load time (see g4e_bake_audio_per_dim_scale),
    // so this single ggml_mul reproduces the HF formula exactly.
    if (L.attn_per_dim_scale) {
        ggml_tensor* scale = ggml_reshape_3d(ctx, L.attn_per_dim_scale, hd, 1, 1);
        Q = ggml_mul(ctx, Q, scale);
    }

    // K constant scale. HF Gemma4AudioAttention L221+L279:
    //   k_scale = log(1 + e) / log(2)        ≈ 1.88539
    //   key_states = key_states * k_scale
    {
        const float k_scale = std::log1p(std::exp(1.0f)) / std::log(2.0f);
        K = ggml_scale(ctx, K, k_scale);
    }

    // Permute to flash-attention layout: (hd, T, n_h)
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Chunked-local attention via mask + logit softcapping. The mask
    // restricts each Q chunk to a local K window; relative position bias
    // (matrix_bd term in HF) is not yet included — that's a follow-up.
    float attn_scale = 1.0f; // per_dim_scale + k_scale replace 1/sqrt(d)
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, attn_mask, attn_scale, 0.0f, hp.attention_logit_cap);
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
    const int N = (int)lhp.num_layers;
    const float eps = lhp.rms_norm_eps;
    const float theta = lhp.rope_theta;
    const int T = n_tokens;
    const int Lk = n_past + T;
    const int ple_dim = 256; // hidden_size_per_layer_input

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // PLE input: token IDs for per-layer embedding lookup. Multimodal
    // tokens (audio frames) carry the audio-soft-token id here so the
    // PLE table is indexed consistently with HF's get_per_layer_inputs.
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

    // ── Per-layer input prep (Gemma4 `project_per_layer_inputs`) ──
    //   ple_lookup = embed_tokens_per_layer(input_ids).reshape(T, n_layers, ple_dim)
    //                * sqrt(ple_dim)                                    # embed_scale on PLE table
    //   ple_proj   = per_layer_model_projection(inputs_embeds)
    //                * (1/sqrt(hidden))                                 # per_layer_model_projection_scale
    //   ple_proj   = per_layer_projection_norm(ple_proj.reshape(T, n_layers, ple_dim))
    //   pli        = (ple_proj + ple_lookup) * (1/sqrt(2))              # per_layer_input_scale
    // The result has shape [ple_dim, n_layers, T] in ggml; each layer
    // picks `pli[:, il, :]` via a strided 2-D view at PLE time below.
    ggml_tensor* per_layer_inputs = nullptr;
    if (ple_ids && m.llm_ple_w) {
        // 1. PLE lookup, scale by sqrt(ple_dim).
        ggml_tensor* ple_lookup = ggml_get_rows(ctx0, m.llm_ple_w, ple_ids); // [n_layers*ple_dim, T]
        ple_lookup = ggml_scale(ctx0, ple_lookup, std::sqrt((float)ple_dim));
        // ggml stores this row-major as ne=[n_layers*ple_dim, T]; reshape
        // to [ple_dim, n_layers, T] so each "layer slice" along ne[1] is
        // contiguous in memory and the per-layer view is just a stride.
        ple_lookup = ggml_reshape_3d(ctx0, ple_lookup, ple_dim, N, T);

        if (m.llm_per_layer_proj_w && m.llm_per_layer_proj_norm) {
            // 2. Linear projection of inputs_embeds, scale by 1/sqrt(hidden).
            ggml_tensor* ple_proj = ggml_mul_mat(ctx0, m.llm_per_layer_proj_w, embeds); // [n_layers*ple_dim, T]
            ple_proj = ggml_scale(ctx0, ple_proj, 1.0f / std::sqrt((float)d));
            ple_proj = ggml_reshape_3d(ctx0, ple_proj, ple_dim, N, T);
            // 3. RMSNorm along ple_dim (ne[0]) — ggml_rms_norm normalises
            //    along the innermost dim by default, which is ple_dim.
            ple_proj = ggml_rms_norm(ctx0, ple_proj, eps);
            ple_proj = ggml_mul(ctx0, ple_proj, m.llm_per_layer_proj_norm); // broadcast [ple_dim] over n_layers, T
            // 4. Mix.
            per_layer_inputs = ggml_add(ctx0, ple_proj, ple_lookup);
            per_layer_inputs = ggml_scale(ctx0, per_layer_inputs, 1.0f / std::sqrt(2.0f));
        } else {
            // Fallback when the projection tensors are missing (older GGUF).
            per_layer_inputs = ple_lookup;
        }
    }

    // Attention params for sliding (local) and full layers. Gemma4's
    // attention uses scaling=1.0 (NOT 1/sqrt(d)) — the q_norm provides
    // the effective scaling. Full-attention layers use partial-rotary
    // RoPE: only the first `partial_rotary_factor * head_dim` entries
    // are rotated. Sliding layers rotate the whole head_dim.
    const int hd_full = (int)lhp.global_head_dim;
    const int n_rot_full = std::max(1, (int)std::round(hd_full * lhp.partial_rotary_factor));

    core_attn::KvSelfAttnParams kvp_local = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)lhp.max_position_embeddings,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ 1.0f, // Gemma4: scaling=1.0 (q_norm replaces 1/sqrt(d))
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };
    kvp_local.v_rms_norm = true; // Gemma4: v_norm = RMSNorm(head_dim, with_scale=False)
    core_attn::KvSelfAttnParams kvp_full = kvp_local;
    kvp_full.head_dim = hd_full;
    kvp_full.rope_theta = lhp.rope_theta_full;
    kvp_full.n_rot = n_rot_full;

    auto is_full_layer = [&](uint32_t il) -> bool {
        return il < lhp.layer_full_mask.size() && lhp.layer_full_mask[il];
    };

    // Per-layer slice of `per_layer_inputs`. Layout is [ple_dim, n_layers, T]
    // so layer `il` maps to rows il*nb[1] (contiguous along ne[0]) repeating
    // every nb[2] bytes per token. A 2-D view captures the (ple_dim, T)
    // submatrix with row stride nb[2].
    auto pli_for_layer = [&](uint32_t il) -> ggml_tensor* {
        if (!per_layer_inputs)
            return nullptr;
        return ggml_view_2d(ctx0, per_layer_inputs, ple_dim, T, per_layer_inputs->nb[2],
                            (size_t)il * per_layer_inputs->nb[1]);
    };

    for (uint32_t il = 0; il < lhp.num_layers; il++) {
        const auto& b = m.llm_layers[il];
        const bool full = is_full_layer(il);
        const auto& kvp = full ? kvp_full : kvp_local;
        ggml_tensor* kv_k_for_layer = (full && ctx->kv_k_full) ? ctx->kv_k_full : ctx->kv_k;
        ggml_tensor* kv_v_for_layer = (full && ctx->kv_v_full) ? ctx->kv_v_full : ctx->kv_v;
        const int donor_il = (il < lhp.kv_share_donor.size()) ? lhp.kv_share_donor[il] : -1;

        // ── Attention block ──
        // HF: residual = h; h = input_layernorm(h); h, _ = self_attn(h, ...);
        //     h = post_attention_layernorm(h); h = residual + h
        ggml_tensor* residual = cur;
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm);

        ggml_tensor* attn;
        if (donor_il < 0) {
            // Non-shared: standard Q/K/V projection + cache write/read.
            // The helper applies q_norm, k_norm, v_norm (RMS-no-weight),
            // RoPE, and writes into the layer's own cache slot.
            attn = core_attn::kv_self_attn(ctx0, gf, x, b.q_proj, b.k_proj, b.v_proj, b.o_proj, b.q_norm, b.k_norm,
                                           positions, (T == 1) ? nullptr : causal_mask, kv_k_for_layer, kv_v_for_layer,
                                           (int)il, n_past, kvp);
        } else {
            // KV-shared layer: compute Q only, read K/V from donor's slot.
            // Donor has the same `layer_type` (sliding/full) by construction,
            // so head_dim, RoPE, n_kv_grp, attn_scale all match `kvp`.
            const int hd_e = kvp.head_dim;
            const int n_q_e = kvp.n_heads;
            const int n_kv_e = kvp.n_kv_heads;
            const int grp_e = kvp.n_kv_grp;
            const int n_rot_e = kvp.n_rot > 0 ? kvp.n_rot : hd_e;

            ggml_tensor* Q = ggml_mul_mat(ctx0, b.q_proj, x);
            Q = ggml_reshape_3d(ctx0, Q, hd_e, n_q_e, T);
            if (b.q_norm) {
                Q = ggml_rms_norm(ctx0, Q, kvp.qk_norm_eps);
                Q = ggml_mul(ctx0, Q, b.q_norm);
            }
            Q = ggml_rope_ext(ctx0, Q, positions, nullptr, n_rot_e, kvp.rope_type, kvp.n_ctx_orig, kvp.rope_theta, 1.0f,
                              0.0f, 1.0f, kvp.rope_beta_fast, kvp.rope_beta_slow);

            // Read full K/V history from donor's slot. K and V are stored
            // post-norm/post-RoPE in the cache, so no extra processing here.
            ggml_tensor* Kfull =
                ggml_cont(ctx0, ggml_view_3d(ctx0, kv_k_for_layer, hd_e, Lk, n_kv_e, kv_k_for_layer->nb[1],
                                             kv_k_for_layer->nb[2], (size_t)donor_il * kv_k_for_layer->nb[3]));
            ggml_tensor* Vfull =
                ggml_cont(ctx0, ggml_view_3d(ctx0, kv_v_for_layer, hd_e, Lk, n_kv_e, kv_v_for_layer->nb[1],
                                             kv_v_for_layer->nb[2], (size_t)donor_il * kv_v_for_layer->nb[3]));

            if (kvp.gqa_mode != core_attn::GQA_NATIVE && grp_e > 1) {
                ggml_tensor* K4 = ggml_reshape_4d(ctx0, Kfull, hd_e, Lk, 1, n_kv_e);
                ggml_tensor* V4 = ggml_reshape_4d(ctx0, Vfull, hd_e, Lk, 1, n_kv_e);
                K4 = ggml_repeat_4d(ctx0, K4, hd_e, Lk, grp_e, n_kv_e);
                V4 = ggml_repeat_4d(ctx0, V4, hd_e, Lk, grp_e, n_kv_e);
                if (kvp.gqa_mode == core_attn::GQA_MANUAL_CONT) {
                    Kfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd_e, Lk, n_q_e));
                    Vfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd_e, Lk, n_q_e));
                } else {
                    Kfull = ggml_reshape_3d(ctx0, K4, hd_e, Lk, n_q_e);
                    Vfull = ggml_reshape_3d(ctx0, V4, hd_e, Lk, n_q_e);
                }
            }

            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            ggml_tensor* attnt = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, (T == 1) ? nullptr : causal_mask,
                                                     kvp.attn_scale, 0.0f, 0.0f);
            attnt = ggml_reshape_2d(ctx0, attnt, hd_e * n_q_e, T);
            attn = ggml_mul_mat(ctx0, b.o_proj, attnt);
        }

        if (b.post_attn_norm) {
            attn = ggml_rms_norm(ctx0, attn, eps);
            attn = ggml_mul(ctx0, attn, b.post_attn_norm);
        }
        cur = ggml_add(ctx0, residual, attn);

        // ── FFN block (SwiGLU) ──
        // HF: residual = h; h = pre_feedforward_layernorm(h); h = mlp(h);
        //     h = post_feedforward_layernorm(h); h = residual + h
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        if (b.pre_ffn_norm)
            x = ggml_mul(ctx0, x, b.pre_ffn_norm);

        // Gemma4TextMLP uses gelu_pytorch_tanh as its gate activation
        // (NOT silu) — see modeling_gemma4.py L1031 + config.json
        // text_config.hidden_activation. ggml_gelu is the tanh-approximation
        // variant matching pytorch's `gelu_pytorch_tanh`.
        ggml_tensor* mlp = core_ffn::geglu(ctx0, x, b.gate_proj, b.up_proj, b.down_proj);

        if (b.post_ffn_norm) {
            mlp = ggml_rms_norm(ctx0, mlp, eps);
            mlp = ggml_mul(ctx0, mlp, b.post_ffn_norm);
        }
        cur = ggml_add(ctx0, residual, mlp);

        // ── PLE block (per-layer-input adjustment) ──
        // HF: residual = h
        //     h = per_layer_input_gate(h)        # Linear 1536→256, no bias
        //     h = act_fn(h)                       # GELU
        //     h = h * per_layer_input             # element-wise (T, 256)
        //     h = per_layer_projection(h)        # Linear 256→1536, no bias
        //     h = post_per_layer_input_norm(h)   # RMSNorm dim=1536
        //     h = residual + h
        // This sits AFTER attention + FFN, BEFORE the layer_scalar multiply.
        if (per_layer_inputs && b.ple_gate && b.ple_proj) {
            ggml_tensor* pli = pli_for_layer(il);
            ggml_tensor* gate = ggml_mul_mat(ctx0, b.ple_gate, cur);    // [ple_dim, T]
            gate = ggml_gelu(ctx0, gate);                               // ACT2FN[hidden_activation] = gelu_pytorch_tanh
            ggml_tensor* gated = ggml_mul(ctx0, gate, pli);             // element-wise [ple_dim, T]
            ggml_tensor* delta = ggml_mul_mat(ctx0, b.ple_proj, gated); // [hidden, T]
            if (b.post_ple_norm) {
                delta = ggml_rms_norm(ctx0, delta, eps);
                delta = ggml_mul(ctx0, delta, b.post_ple_norm);
            }
            cur = ggml_add(ctx0, cur, delta);
        }

        // ── layer_scalar (single multiply at end of layer) ──
        // HF: hidden_states *= self.layer_scalar
        if (b.layer_scalar)
            cur = ggml_mul(ctx0, cur, b.layer_scalar);
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

    // PLE token IDs. Caller (transcribe / embed_tokens) is responsible for
    // sizing ctx->ple_token_ids to match n_tokens before we get here. For
    // audio frames we use audio_soft_token_id; for text tokens we use the
    // actual token id. Mismatched length → fall back to bos_id repeats.
    if ((int)ctx->ple_token_ids.size() != n_tokens) {
        if (ctx->verbosity >= 2)
            fprintf(stderr, "gemma4_e2b: ple_token_ids size %zu != n_tokens %d, padding with bos\n",
                    ctx->ple_token_ids.size(), n_tokens);
        ctx->ple_token_ids.assign((size_t)n_tokens, ctx->bos_id);
    }

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
    ggml_tensor* ple_ids_in = ggml_graph_get_tensor(gf, "ple_ids");
    if (ple_ids_in)
        ggml_backend_tensor_set(ple_ids_in, ctx->ple_token_ids.data(), 0, (size_t)n_tokens * sizeof(int32_t));
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

    // Stash for the next g4e_run_llm_kv call so the PLE block has the
    // correct per-token ids. The decode loop takes this branch with n=1
    // for every step; prefill takes a different path (transcribe builds
    // the full ple_token_ids vector explicitly, including audio
    // placeholders, before the prefill forward).
    ctx->ple_token_ids.assign(ids, ids + n);

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
static bool g4e_gguf_bool(gguf_context* ctx, const char* key, bool def = false) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_bool(ctx, id) : def;
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
    lhp.global_head_dim = g4e_gguf_u32(gctx, "gemma4e2b.llm.global_head_dim", lhp.head_dim);
    lhp.intermediate_size = g4e_gguf_u32(gctx, "gemma4e2b.llm.intermediate_size", 6144);
    lhp.vocab_size = g4e_gguf_u32(gctx, "gemma4e2b.llm.vocab_size", 262144);
    lhp.max_position_embeddings = g4e_gguf_u32(gctx, "gemma4e2b.llm.max_position_embeddings", 131072);
    lhp.sliding_window = g4e_gguf_u32(gctx, "gemma4e2b.llm.sliding_window", 512);
    lhp.num_kv_shared_layers = g4e_gguf_u32(gctx, "gemma4e2b.llm.num_kv_shared_layers", 0);
    lhp.rope_theta = g4e_gguf_f32(gctx, "gemma4e2b.llm.rope_theta", 10000.0f);
    lhp.rope_theta_full = g4e_gguf_f32(gctx, "gemma4e2b.llm.rope_theta_full", 1000000.0f);
    lhp.partial_rotary_factor = g4e_gguf_f32(gctx, "gemma4e2b.llm.partial_rotary_factor", 0.25f);
    lhp.final_logit_softcapping = g4e_gguf_f32(gctx, "gemma4e2b.llm.final_logit_softcapping", 30.0f);
    lhp.rms_norm_eps = g4e_gguf_f32(gctx, "gemma4e2b.llm.rms_norm_eps", 1e-6f);
    lhp.use_double_wide_mlp = g4e_gguf_bool(gctx, "gemma4e2b.llm.use_double_wide_mlp", false);
    lhp.attention_k_eq_v = g4e_gguf_bool(gctx, "gemma4e2b.llm.attention_k_eq_v", false);

    // Per-layer attention type mask: 1 = full, 0 = sliding. New
    // converter persists this directly. Older GGUFs (pre-2026-04-28
    // converter) didn't, in which case we infer it from tensor shapes
    // after the weights load.
    {
        const int mask_key = gguf_find_key(gctx, "gemma4e2b.llm.layer_full_mask");
        const int n_layers = (int)lhp.num_layers;
        lhp.layer_full_mask.assign(n_layers, 0);
        if (mask_key >= 0) {
            const int n_arr = gguf_get_arr_n(gctx, mask_key);
            const auto* arr_data = (const int32_t*)gguf_get_arr_data(gctx, mask_key);
            const int n_take = std::min(n_arr, n_layers);
            for (int i = 0; i < n_take; i++)
                lhp.layer_full_mask[i] = arr_data[i] ? 1 : 0;
        }
    }

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

    // Helper to read a 1-element F32 scalar from the GGUF (for QAT
    // clipping bounds). Returns the default when the tensor is absent.
    auto read_clip_scalar = [&](ggml_tensor* t, float def) -> float {
        if (!t)
            return def;
        float v = def;
        ggml_backend_tensor_get(t, &v, 0, sizeof(float));
        return v;
    };

    m.audio_layers.resize(ahp.num_layers);
    for (uint32_t i = 0; i < ahp.num_layers; i++) {
        char buf[128];
        auto& L = m.audio_layers[i];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "audio.layers.%u.%s", i, suffix);
            return get(buf);
        };
        auto load_clip = [&](const char* prefix, g4e_clip& c) {
            char k[128];
            snprintf(k, sizeof(k), "%s.input_min", prefix);
            c.in_min = read_clip_scalar(g(k), -INFINITY);
            snprintf(k, sizeof(k), "%s.input_max", prefix);
            c.in_max = read_clip_scalar(g(k), INFINITY);
            snprintf(k, sizeof(k), "%s.output_min", prefix);
            c.out_min = read_clip_scalar(g(k), -INFINITY);
            snprintf(k, sizeof(k), "%s.output_max", prefix);
            c.out_max = read_clip_scalar(g(k), INFINITY);
        };
        // Macaron FFN 1
        L.ffn1_pre_ln = g("ffn1.pre_ln.weight");
        L.ffn1_up_w = g("ffn1.up.weight");
        L.ffn1_down_w = g("ffn1.down.weight");
        L.ffn1_post_ln = g("ffn1.post_ln.weight");
        load_clip("ffn1.up", L.clip_ffn1_up);
        load_clip("ffn1.down", L.clip_ffn1_down);
        // Self-attention
        L.attn_pre_ln = g("attn_pre_ln.weight");
        L.attn_q_w = g("attn.q.weight");
        L.attn_k_w = g("attn.k.weight");
        L.attn_v_w = g("attn.v.weight");
        L.attn_o_w = g("attn.o.weight");
        L.attn_per_dim_scale = g("attn.per_dim_scale");
        L.attn_rel_k_w = g("attn.rel_k.weight");
        L.attn_post_ln = g("attn_post_ln.weight");
        load_clip("attn.q", L.clip_q);
        load_clip("attn.k", L.clip_k);
        load_clip("attn.v", L.clip_v);
        load_clip("attn.o", L.clip_o);
        // LightConv1d
        L.conv_pre_ln = g("conv.pre_ln.weight");
        L.conv_gate_w = g("conv.gate_proj.weight");
        L.conv_dw_w = g("conv.dw_conv.weight");
        L.conv_ln = g("conv.conv_ln.weight");
        L.conv_out_w = g("conv.out_proj.weight");
        load_clip("conv.gate_proj", L.clip_conv_gate);
        load_clip("conv.out_proj", L.clip_conv_out);
        // Macaron FFN 2
        L.ffn2_pre_ln = g("ffn2.pre_ln.weight");
        L.ffn2_up_w = g("ffn2.up.weight");
        L.ffn2_down_w = g("ffn2.down.weight");
        L.ffn2_post_ln = g("ffn2.post_ln.weight");
        load_clip("ffn2.up", L.clip_ffn2_up);
        load_clip("ffn2.down", L.clip_ffn2_down);
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
    m.llm_per_layer_proj_w = get("llm.per_layer_model_projection.weight");
    m.llm_per_layer_proj_norm = get("llm.per_layer_projection_norm.weight");

    m.llm_layers.resize(lhp.num_layers);
    for (uint32_t i = 0; i < lhp.num_layers; i++) {
        char buf[128];
        auto& L = m.llm_layers[i];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.layers.%u.%s", i, suffix);
            return get(buf);
        };
        L.attn_norm = g("attn_norm.weight");
        // Converter renames: q_proj/k_proj/v_proj → q/k/v; o_proj is
        // kept as o_proj. Try the short names first; fall back to the
        // long *_proj names if a future converter reverses the rename.
        L.q_proj = g("attn.q.weight");
        if (!L.q_proj)
            L.q_proj = g("attn.q_proj.weight");
        L.k_proj = g("attn.k.weight");
        if (!L.k_proj)
            L.k_proj = g("attn.k_proj.weight");
        L.v_proj = g("attn.v.weight");
        if (!L.v_proj)
            L.v_proj = g("attn.v_proj.weight");
        L.o_proj = g("attn.o_proj.weight");
        if (!L.o_proj)
            L.o_proj = g("attn.o.weight");
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

    // Infer layer_full_mask + global_head_dim from tensor shapes when
    // the GGUF didn't persist them. Older converters wrote neither;
    // the runtime can recover both from q.weight->ne[1] (which equals
    // n_heads * head_dim_for_this_layer). Uniform shape across layers
    // means the model only has one attention type → leave mask all-0.
    {
        int from_metadata = 0;
        for (int v : lhp.layer_full_mask)
            from_metadata += v;
        if (from_metadata == 0) {
            // Discover per-layer q.ne[1] values.
            int min_cols = INT32_MAX, max_cols = 0;
            for (uint32_t il = 0; il < lhp.num_layers; il++) {
                ggml_tensor* q = m.llm_layers[il].q_proj;
                if (!q)
                    continue;
                int cols = (int)q->ne[1];
                if (cols < min_cols)
                    min_cols = cols;
                if (cols > max_cols)
                    max_cols = cols;
            }
            if (min_cols != INT32_MAX && max_cols > min_cols) {
                // Two distinct sizes — the bigger one is full attention.
                const uint32_t inferred_local_hd = (uint32_t)(min_cols / lhp.num_heads);
                const uint32_t inferred_global_hd = (uint32_t)(max_cols / lhp.num_heads);
                if (lhp.global_head_dim == lhp.head_dim) {
                    // Metadata only had a single head_dim; trust shapes.
                    lhp.head_dim = inferred_local_hd;
                    lhp.global_head_dim = inferred_global_hd;
                }
                int n_full = 0;
                for (uint32_t il = 0; il < lhp.num_layers; il++) {
                    ggml_tensor* q = m.llm_layers[il].q_proj;
                    if (!q)
                        continue;
                    if ((int)q->ne[1] == max_cols) {
                        lhp.layer_full_mask[il] = 1;
                        n_full++;
                    }
                }
                fprintf(stderr,
                        "gemma4_e2b: inferred layer_full_mask: %d full / %u total "
                        "(local head_dim=%u, full head_dim=%u)\n",
                        n_full, lhp.num_layers, lhp.head_dim, lhp.global_head_dim);
            }
        }
    }

    // Bake `q_scale * softplus(per_dim_scale)` into the audio attention's
    // per_dim_scale tensor (HF Gemma4AudioAttention applies this every
    // forward pass; we do it once at load).
    g4e_bake_audio_per_dim_scale(m);

    // KV-share donor map (Gemma4: last `num_kv_shared_layers` layers reuse
    // K/V from the last earlier layer of the same `layer_type`).
    g4e_compute_kv_share_donor(lhp);
    if (params.verbosity >= 1 && lhp.num_kv_shared_layers > 0) {
        int n_shared = 0;
        for (int v : lhp.kv_share_donor)
            if (v >= 0)
                n_shared++;
        fprintf(stderr, "gemma4_e2b: kv-share %d/%u layers reuse donor K/V\n", n_shared, lhp.num_layers);
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
    // Gemma4 mel: fft_length=512, frame_length=320, hop=160. Hann window
    // is 320 samples long (gets center-padded to 512 inside core_mel).
    // Filterbank is sized to n_fft=512 (n_freqs=257) over 0..8000 Hz.
    g4e_gen_hann_window(320, ctx->mel_window);
    g4e_gen_mel_filterbank(128, 512, 16000, ctx->mel_filterbank);

    // Look up special token IDs from vocab. Gemma4 uses `<|turn>` /
    // `<turn|>` for turn markers (not `<start_of_turn>` / `<end_of_turn>`
    // like Gemma2/3) and `<|audio|>` as the audio soft-token placeholder.
    // We accept both naming styles so the loader works on a Gemma2/3-
    // converted GGUF too.
    for (int i = 0; i < (int)m.vocab.size(); i++) {
        if (m.vocab[i] == "<pad>")
            ctx->pad_id = i;
        else if (m.vocab[i] == "<bos>")
            ctx->bos_id = i;
        else if (m.vocab[i] == "<eos>" || m.vocab[i] == "</s>")
            ctx->eos_id = i;
        else if (m.vocab[i] == "<|turn>" || m.vocab[i] == "<start_of_turn>")
            ctx->start_of_turn_id = i;
        else if (m.vocab[i] == "<turn|>" || m.vocab[i] == "<end_of_turn>")
            ctx->end_of_turn_id = i;
        else if (m.vocab[i] == "<|audio|>" || m.vocab[i] == "<audio_soft_token>" || m.vocab[i] == "<audio>")
            ctx->audio_soft_token_id = i;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "gemma4_e2b: audio %uL×%u, llm %uL×%u, vocab %u\n", ahp.num_layers, ahp.hidden_size,
                lhp.num_layers, lhp.hidden_size, lhp.vocab_size);
        fprintf(stderr, "gemma4_e2b: bos=%d eos=%d start_of_turn=%d end_of_turn=%d audio_soft=%d\n", ctx->bos_id,
                ctx->eos_id, ctx->start_of_turn_id, ctx->end_of_turn_id, ctx->audio_soft_token_id);
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

    // ── Step 1: Mel spectrogram ──
    // Gemma4AudioFeatureExtractor params (processor_config.json):
    //   feature_size: 128       n_mels
    //   fft_length: 512         FFT size (zero-padded)
    //   frame_length: 320       Hann window length (<= n_fft)
    //   hop_length: 160         frame stride
    //   sampling_rate: 16000
    //   mel_floor: 0.001        clamp before log
    //   max_frequency: 8000
    const int n_fft = 512, hop = 160, n_mels = 128;
    const int win_length = 320;
    const int n_freqs = n_fft / 2 + 1;

    // HF-faithful mel: bypasses core_mel because Gemma4's FE uses
    // semicausal padding + unfold(size=frame_length+1) which doesn't fit
    // core_mel's API. See g4e_compute_mel_hf_faithful for details.
    const float* hann_ptr = ctx->mel_window.data();     // length win_length
    const float* filt_ptr = ctx->mel_filterbank.data(); // (n_freqs, n_mels) FreqsMels layout

    int T_mel = 0;
    const float mel_floor = 0.001f; // HF default
    auto mel = g4e_compute_mel_hf_faithful(pcm, n_samples, n_fft, win_length, hop, n_mels, hann_ptr, filt_ptr,
                                           mel_floor, T_mel);
    if (mel.empty()) {
        fprintf(stderr, "gemma4_e2b: mel computation failed\n");
        return nullptr;
    }
    // Cap to 30s
    if (T_mel > 3000)
        T_mel = 3000;
    (void)n_freqs;

    if (verbose)
        fprintf(stderr, "gemma4_e2b: mel %dx%d (%.1f ms)\n", n_mels, T_mel, (ggml_time_us() - t0) / 1000.0);

    // ── Step 2-4: Encoder (Conv2D sub + Conformer + output proj) ────────
    int64_t t_enc0 = ggml_time_us();

    // Build single encoder graph: mel → conv2d sub → 12 conformer layers → output proj
    size_t enc_mem = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(131072, false);
    std::vector<uint8_t> enc_meta(enc_mem);
    ggml_init_params enc_ip = {enc_mem, enc_meta.data(), true};
    ggml_context* ectx = ggml_init(enc_ip);
    ggml_cgraph* enc_gf = ggml_new_graph_custom(ectx, 131072, false);

    // Input: mel [T_mel, n_mels, 1, 1] for conv2d
    ggml_tensor* mel_in = ggml_new_tensor_4d(ectx, GGML_TYPE_F32, n_mels, T_mel, 1, 1);
    ggml_set_name(mel_in, "mel");
    ggml_set_input(mel_in);

    // Conv2D subsampling layer 0: Conv2d(1→128, k=3, s=2, p=1) + LayerNorm + ReLU
    // HF Gemma4AudioSubSampleConvProjectionLayer uses nn.LayerNorm (NOT
    // RMSNorm) and nn.ReLU (NOT SiLU). The norm is over the channel dim
    // with `elementwise_affine=True, bias=False`, so weight only — same
    // shape as RMSNorm but with mean-centering.
    ggml_tensor* h = mel_in;
    if (m.sub_conv0_w) {
        h = ggml_conv_2d(ectx, m.sub_conv0_w, h, 2, 2, 1, 1, 1, 1);
        // h: [OW, OH, 128, 1] where OW≈T_mel/2, OH≈n_mels/2=64
        if (m.sub_norm0_w) {
            int ow = (int)h->ne[0], oh = (int)h->ne[1], c = (int)h->ne[2];
            h = ggml_reshape_2d(ectx, h, ow * oh, c);
            h = ggml_cont(ectx, ggml_transpose(ectx, h)); // [c, ow*oh]
            h = ggml_norm(ectx, h, eps);                  // LayerNorm (mean+var)
            h = ggml_mul(ectx, h, m.sub_norm0_w);
            h = ggml_cont(ectx, ggml_transpose(ectx, h)); // [ow*oh, c]
            h = ggml_reshape_4d(ectx, h, ow, oh, c, 1);
        }
        h = ggml_relu(ectx, h);
    }

    // Conv2D subsampling layer 1: Conv2d(128→32, k=3, s=2, p=1) + LayerNorm + ReLU
    if (m.sub_conv1_w) {
        h = ggml_conv_2d(ectx, m.sub_conv1_w, h, 2, 2, 1, 1, 1, 1);
        // h: [OW2, OH2, 32, 1] where OW2≈T_mel/4, OH2≈n_mels/4=32
        if (m.sub_norm1_w) {
            int ow = (int)h->ne[0], oh = (int)h->ne[1], c = (int)h->ne[2];
            h = ggml_reshape_2d(ectx, h, ow * oh, c);
            h = ggml_cont(ectx, ggml_transpose(ectx, h));
            h = ggml_norm(ectx, h, eps);
            h = ggml_mul(ectx, h, m.sub_norm1_w);
            h = ggml_cont(ectx, ggml_transpose(ectx, h));
            h = ggml_reshape_4d(ectx, h, ow, oh, c, 1);
        }
        h = ggml_relu(ectx, h);
    }

    // After two stride-2 convs, h has ggml ne=(M_sub, T_sub, C, 1).
    // HF flattens (B, C, T, M) → (B, T, M, C) → (B, T, M*C) with C-fast.
    // ggml_permute(a, p0, p1, p2, p3): pN = OUTPUT axis for INPUT axis N.
    int T_sub = (int)h->ne[1];
    int M_sub = (int)h->ne[0];
    int C_out = (int)h->ne[2];
    // M (input 0) → output 1; T (input 1) → output 2; C (input 2) → output 0.
    h = ggml_cont(ectx, ggml_permute(ectx, h, 1, 2, 0, 3)); // (C, M, T, 1)
    int feat_dim = C_out * M_sub;
    h = ggml_reshape_2d(ectx, h, feat_dim, T_sub);

    // Input projection: Linear(1024→1024)
    if (m.sub_input_proj_w) {
        h = ggml_mul_mat(ectx, m.sub_input_proj_w, h); // (1024, T_sub)
    }

    int hidden = (int)ahp.hidden_size;

    // Audio attention inputs: pos_enc (constant) + per-block boundary mask.
    const int chunk_size_a = (int)ahp.chunk_size;
    const int max_past_a = (int)ahp.context_left - 1;
    const int n_pos_a = chunk_size_a + 1;
    const int context_size_a = chunk_size_a + max_past_a;
    const int num_blocks_a = (T_sub + chunk_size_a - 1) / chunk_size_a;
    ggml_tensor* audio_pos_enc = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, hidden, n_pos_a);
    ggml_set_name(audio_pos_enc, "audio_pos_enc");
    ggml_set_input(audio_pos_enc);
    ggml_tensor* audio_pad_mask = ggml_new_tensor_2d(ectx, GGML_TYPE_F16, context_size_a, num_blocks_a);
    ggml_set_name(audio_pad_mask, "audio_pad_mask");
    ggml_set_input(audio_pad_mask);

    // ── Conformer encoder (12 layers) ──
    for (uint32_t il = 0; il < ahp.num_layers; il++) {
        const auto& L = m.audio_layers[il];

        // Macaron FFN 1 (half-step)
        if (L.ffn1_up_w)
            h = build_macaron_ffn(ectx, h, L.ffn1_pre_ln, L.ffn1_up_w, L.ffn1_down_w, L.ffn1_post_ln,
                                  ahp.residual_weight, eps, L.clip_ffn1_up, L.clip_ffn1_down);

        // Self-attention (chunked-local + relative position bias).
        h = build_conformer_self_attn(ectx, h, L, ahp, audio_pos_enc, audio_pad_mask);

        // LightConv1d
        if (L.conv_gate_w)
            h = build_light_conv1d(ectx, h, L, T_sub, hidden, eps);

        // Macaron FFN 2 (half-step)
        if (L.ffn2_up_w)
            h = build_macaron_ffn(ectx, h, L.ffn2_pre_ln, L.ffn2_up_w, L.ffn2_down_w, L.ffn2_post_ln,
                                  ahp.residual_weight, eps, L.clip_ffn2_up, L.clip_ffn2_down);

        // Output layer norm
        if (L.out_ln) {
            h = ggml_rms_norm(ectx, h, eps);
            h = ggml_mul(ectx, h, L.out_ln);
        }
    }

    // ── Output projection: Linear(1024→1536, bias) + adapter ──
    if (m.audio_output_proj_w) {
        h = ggml_mul_mat(ectx, m.audio_output_proj_w, h);
        if (m.audio_output_proj_b)
            h = ggml_add(ectx, h, m.audio_output_proj_b);
    }
    // Audio→LLM adapter (HF Gemma4MultimodalEmbedder.forward L1973-1974):
    //   embs_normed = embedding_pre_projection_norm(inputs_embeds)  # RMSNorm(no-weight)
    //   return embedding_projection(embs_normed)                     # Linear(no-bias)
    if (m.audio_embed_proj_w) {
        h = ggml_rms_norm(ectx, h, eps); // pre-projection norm, no weight
        h = ggml_mul_mat(ectx, m.audio_embed_proj_w, h);
    }

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

    // Audio attention inputs: pos_enc (constant per encoder, shape
    // (hidden, n_pos)) and per-block pad_mask (shape (context_size, num_blocks)).
    {
        ggml_tensor* pe = ggml_graph_get_tensor(enc_gf, "audio_pos_enc");
        if (pe) {
            const int hidden_a = (int)pe->ne[0];
            const int chunk_a = (int)ahp.chunk_size;
            auto pe_buf = g4e_make_audio_pos_enc(hidden_a, chunk_a);
            ggml_backend_tensor_set(pe, pe_buf.data(), 0, pe_buf.size() * sizeof(float));
        }
        ggml_tensor* pm = ggml_graph_get_tensor(enc_gf, "audio_pad_mask");
        if (pm) {
            const int context_size_a = (int)pm->ne[0];
            const int num_blocks_a = (int)pm->ne[1];
            const int chunk_a = (int)ahp.chunk_size;
            const int max_past_a = (int)ahp.context_left - 1;
            const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> pm_buf((size_t)context_size_a * num_blocks_a);
            for (int b = 0; b < num_blocks_a; b++) {
                for (int k = 0; k < context_size_a; k++) {
                    int abs_k = b * chunk_a - max_past_a + k;
                    pm_buf[(size_t)b * context_size_a + k] = (abs_k >= 0 && abs_k < T_sub) ? zero_h : neginf_h;
                }
            }
            ggml_backend_tensor_set(pm, pm_buf.data(), 0, pm_buf.size() * sizeof(ggml_fp16_t));
        }
    }

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

    // Optional audio-span markers. Gemma4 wraps audio embeddings with
    // `<|audio>` (boa) and `<audio|>` (eoa); the soft tokens between
    // them are the placeholders for the audio frames.
    int boa_id = find_token("<|audio>");
    int eoa_id = find_token("<audio|>");
    if (boa_id >= 0)
        prompt_ids.push_back(boa_id);

    int audio_insert_pos = (int)prompt_ids.size(); // audio goes here

    // "Transcribe the following audio clip into text."
    // Simple approach: try to find common subword tokens
    std::vector<std::string> text_tokens = {"Transcribe", " the",  " following", " audio",
                                            " clip",      " into", " text",      "."};
    if (eoa_id >= 0) {
        // Push eoa AFTER the audio soft tokens — but the soft tokens are
        // injected later via the prefix/suffix split, so we just remember
        // to start the suffix with eoa. Easiest: push it now into the
        // suffix-portion of prompt_ids (positions ≥ audio_insert_pos).
        prompt_ids.push_back(eoa_id);
    }
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

    // Per-token IDs for the prefill PLE lookup. HF Gemma4Model.forward
    // (line 2208-2221) replaces multimodal token ids with `pad_token_id`
    // BEFORE the `embed_tokens_per_layer` lookup, then constructs the
    // per_layer_inputs from those pad-replaced ids. So at audio
    // positions we feed pad_id (0) into the lookup, NOT the audio
    // soft-token id — even though the audio soft-token is what the
    // tokenizer produces in the input ids before merging.
    {
        ctx->ple_token_ids.clear();
        ctx->ple_token_ids.reserve((size_t)T_total);
        for (int i = 0; i < n_prefix; i++)
            ctx->ple_token_ids.push_back(prompt_ids[i]);
        for (int i = 0; i < N_audio; i++)
            ctx->ple_token_ids.push_back(ctx->pad_id);
        for (int i = 0; i < n_suffix; i++)
            ctx->ple_token_ids.push_back(prompt_ids[(size_t)n_prefix + i]);
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
    cfg.eos_id = ctx->end_of_turn_id >= 0 ? ctx->end_of_turn_id : ctx->eos_id;
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

// ── Stage hooks for crispasr-diff ──────────────────────────────────────────
// These mirror the parakeet/voxtral pattern: each one runs a slice of the
// pipeline and returns a malloc'd float buffer the caller frees. They share
// the mel + encoder code path used by gemma4_e2b_transcribe so any number
// drift between the diff harness and end-to-end runs would be a bug here.

extern "C" float* gemma4_e2b_compute_mel(struct gemma4_e2b_context* ctx, const float* pcm, int n_samples,
                                         int* out_n_mels, int* out_T_mel) {
    if (!ctx || !pcm || n_samples <= 0 || !out_n_mels || !out_T_mel)
        return nullptr;

    // Match Gemma4AudioFeatureExtractor exactly via g4e_compute_mel_hf_faithful.
    const int n_fft = 512, hop = 160, n_mels = 128, win_length = 320;
    const float* hann_ptr = ctx->mel_window.data();     // length win_length
    const float* filt_ptr = ctx->mel_filterbank.data(); // (n_freqs, n_mels) FreqsMels

    int T_mel = 0;
    auto mel = g4e_compute_mel_hf_faithful(pcm, n_samples, n_fft, win_length, hop, n_mels, hann_ptr, filt_ptr,
                                           /*mel_floor=*/0.001f, T_mel);
    if (mel.empty())
        return nullptr;
    if (T_mel > 3000)
        T_mel = 3000;

    // mel is already in (T_mel, n_mels) row-major (matches HF FE's
    // natural output layout). Pass through directly.
    const size_t n = (size_t)n_mels * (size_t)T_mel;
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, mel.data(), std::min((size_t)mel.size(), n) * sizeof(float));
    *out_n_mels = n_mels;
    *out_T_mel = T_mel;
    return out;
}

extern "C" float* gemma4_e2b_run_encoder(struct gemma4_e2b_context* ctx, const float* mel, int n_mels, int T_mel,
                                         int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || n_mels <= 0 || T_mel <= 0 || !out_T_enc || !out_d_model)
        return nullptr;

    auto& m = ctx->model;
    auto& ahp = m.audio_hp;
    const float eps = m.llm_hp.rms_norm_eps;

    // Build encoder graph (same shape as gemma4_e2b_transcribe).
    size_t enc_mem = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(131072, false);
    std::vector<uint8_t> enc_meta(enc_mem);
    ggml_init_params enc_ip = {enc_mem, enc_meta.data(), true};
    ggml_context* ectx = ggml_init(enc_ip);
    ggml_cgraph* enc_gf = ggml_new_graph_custom(ectx, 131072, false);

    ggml_tensor* mel_in = ggml_new_tensor_4d(ectx, GGML_TYPE_F32, n_mels, T_mel, 1, 1);
    ggml_set_name(mel_in, "mel");
    ggml_set_input(mel_in);

    ggml_tensor* h = mel_in;
    if (m.sub_conv0_w) {
        h = ggml_conv_2d(ectx, m.sub_conv0_w, h, 2, 2, 1, 1, 1, 1);
        if (m.sub_norm0_w) {
            int ow = (int)h->ne[0], oh = (int)h->ne[1], c = (int)h->ne[2];
            h = ggml_reshape_2d(ectx, h, ow * oh, c);
            h = ggml_cont(ectx, ggml_transpose(ectx, h));
            h = ggml_norm(ectx, h, eps);
            h = ggml_mul(ectx, h, m.sub_norm0_w);
            h = ggml_cont(ectx, ggml_transpose(ectx, h));
            h = ggml_reshape_4d(ectx, h, ow, oh, c, 1);
        }
        h = ggml_relu(ectx, h);
    }
    if (m.sub_conv1_w) {
        h = ggml_conv_2d(ectx, m.sub_conv1_w, h, 2, 2, 1, 1, 1, 1);
        if (m.sub_norm1_w) {
            int ow = (int)h->ne[0], oh = (int)h->ne[1], c = (int)h->ne[2];
            h = ggml_reshape_2d(ectx, h, ow * oh, c);
            h = ggml_cont(ectx, ggml_transpose(ectx, h));
            h = ggml_norm(ectx, h, eps);
            h = ggml_mul(ectx, h, m.sub_norm1_w);
            h = ggml_cont(ectx, ggml_transpose(ectx, h));
            h = ggml_reshape_4d(ectx, h, ow, oh, c, 1);
        }
        h = ggml_relu(ectx, h);
    }

    // h ne=(M_sub, T_sub, C, 1). Flatten (M, T, C) → (C, M, T) for HF's
    // C-fast, M-slow per-frame feature ordering, then reshape (C*M, T).
    // ggml_permute(a, p0, p1, p2, p3): pN = OUTPUT axis for INPUT axis N.
    //   M (input 0) → output 1
    //   T (input 1) → output 2
    //   C (input 2) → output 0
    int T_sub = (int)h->ne[1];
    int M_sub = (int)h->ne[0];
    int C_out = (int)h->ne[2];
    h = ggml_cont(ectx, ggml_permute(ectx, h, 1, 2, 0, 3)); // (C, M, T, 1)
    int feat_dim = C_out * M_sub;
    h = ggml_reshape_2d(ectx, h, feat_dim, T_sub);
    if (m.sub_input_proj_w)
        h = ggml_mul_mat(ectx, m.sub_input_proj_w, h);

    ggml_set_name(h, "audio_subsample_output");
    ggml_set_output(h);

    int hidden = (int)ahp.hidden_size;
    const int chunk_size_a = (int)ahp.chunk_size;
    const int max_past_a = (int)ahp.context_left - 1;
    const int n_pos_a = chunk_size_a + 1;
    const int context_size_a = chunk_size_a + max_past_a;
    const int num_blocks_a = (T_sub + chunk_size_a - 1) / chunk_size_a;
    ggml_tensor* audio_pos_enc = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, hidden, n_pos_a);
    ggml_set_name(audio_pos_enc, "audio_pos_enc");
    ggml_set_input(audio_pos_enc);
    ggml_tensor* audio_pad_mask = ggml_new_tensor_2d(ectx, GGML_TYPE_F16, context_size_a, num_blocks_a);
    ggml_set_name(audio_pad_mask, "audio_pad_mask");
    ggml_set_input(audio_pad_mask);

    for (uint32_t il = 0; il < ahp.num_layers; il++) {
        const auto& L = m.audio_layers[il];
        if (L.ffn1_up_w)
            h = build_macaron_ffn(ectx, h, L.ffn1_pre_ln, L.ffn1_up_w, L.ffn1_down_w, L.ffn1_post_ln,
                                  ahp.residual_weight, eps, L.clip_ffn1_up, L.clip_ffn1_down);
        h = build_conformer_self_attn(ectx, h, L, ahp, audio_pos_enc, audio_pad_mask);
        if (L.conv_gate_w)
            h = build_light_conv1d(ectx, h, L, T_sub, hidden, eps);
        if (L.ffn2_up_w)
            h = build_macaron_ffn(ectx, h, L.ffn2_pre_ln, L.ffn2_up_w, L.ffn2_down_w, L.ffn2_post_ln,
                                  ahp.residual_weight, eps, L.clip_ffn2_up, L.clip_ffn2_down);
        if (L.out_ln) {
            h = ggml_rms_norm(ectx, h, eps);
            h = ggml_mul(ectx, h, L.out_ln);
        }
        // Per-layer stage tap. Names align with the HF reference dump's
        // forward-hook on audio_tower.layers[il].
        char layer_name[64];
        snprintf(layer_name, sizeof(layer_name), "audio_layer_%u", il);
        ggml_set_name(h, layer_name);
        ggml_set_output(h);
    }
    if (m.audio_output_proj_w) {
        h = ggml_mul_mat(ectx, m.audio_output_proj_w, h);
        if (m.audio_output_proj_b)
            h = ggml_add(ectx, h, m.audio_output_proj_b);
    }
    // Stage tap: post-output_proj, pre-adapter. Matches HF
    // audio_tower.last_hidden_state.
    ggml_set_name(h, "audio_tower_output");
    ggml_set_output(h);

    // Audio→LLM adapter: RMSNorm(no-weight) then Linear(no-bias)
    // matching HF Gemma4MultimodalEmbedder.forward L1973-1974.
    if (m.audio_embed_proj_w) {
        h = ggml_rms_norm(ectx, h, eps);
        h = ggml_mul_mat(ectx, m.audio_embed_proj_w, h);
    }

    ggml_set_name(h, "encoder_out");
    ggml_set_output(h);
    ggml_build_forward_expand(enc_gf, h);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, enc_gf)) {
        ggml_free(ectx);
        return nullptr;
    }
    ggml_tensor* mel_t = ggml_graph_get_tensor(enc_gf, "mel");
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)T_mel * n_mels * sizeof(float));
    {
        ggml_tensor* pe = ggml_graph_get_tensor(enc_gf, "audio_pos_enc");
        if (pe) {
            const int hidden_a = (int)pe->ne[0];
            const int chunk_a = (int)ahp.chunk_size;
            auto pe_buf = g4e_make_audio_pos_enc(hidden_a, chunk_a);
            ggml_backend_tensor_set(pe, pe_buf.data(), 0, pe_buf.size() * sizeof(float));
        }
        ggml_tensor* pm = ggml_graph_get_tensor(enc_gf, "audio_pad_mask");
        if (pm) {
            const int context_size_a = (int)pm->ne[0];
            const int num_blocks_a = (int)pm->ne[1];
            const int chunk_a = (int)ahp.chunk_size;
            const int max_past_a = (int)ahp.context_left - 1;
            const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> pm_buf((size_t)context_size_a * num_blocks_a);
            for (int b = 0; b < num_blocks_a; b++) {
                for (int k = 0; k < context_size_a; k++) {
                    int abs_k = b * chunk_a - max_past_a + k;
                    pm_buf[(size_t)b * context_size_a + k] = (abs_k >= 0 && abs_k < T_sub) ? zero_h : neginf_h;
                }
            }
            ggml_backend_tensor_set(pm, pm_buf.data(), 0, pm_buf.size() * sizeof(ggml_fp16_t));
        }
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, enc_gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ectx);
        return nullptr;
    }

    ggml_tensor* enc_out_t = ggml_graph_get_tensor(enc_gf, "encoder_out");
    int proj_dim = (int)enc_out_t->ne[0];
    int N_audio = (int)enc_out_t->ne[1];
    const size_t n_floats = (size_t)proj_dim * (size_t)N_audio;
    float* out = (float*)malloc(n_floats * sizeof(float));
    if (!out) {
        ggml_free(ectx);
        return nullptr;
    }
    ggml_backend_tensor_get(enc_out_t, out, 0, n_floats * sizeof(float));

    // Optional intermediate-stage dump for the diff harness. When
    // CRISPASR_DUMP_DIR is set, write each named stage (audio_subsample
    // _output, audio_layer_<il>, audio_tower_output) to <dir>/<name>.bin
    // as raw F32. The diff harness then compares per-stage cos against
    // the python reference.
    if (const char* dump_dir = std::getenv("CRISPASR_DUMP_DIR")) {
        auto dump = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(enc_gf, name);
            if (!t)
                return;
            const size_t nb = ggml_nbytes(t);
            std::vector<uint8_t> buf(nb);
            ggml_backend_tensor_get(t, buf.data(), 0, nb);
            char path[512];
            snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, name);
            FILE* f = fopen(path, "wb");
            if (f) {
                // Header: magic 'G4DR', n_dims, ne[0..n_dims-1] as int32.
                int32_t hdr[6] = {0x52443447,       0, (int32_t)t->ne[0], (int32_t)t->ne[1], (int32_t)t->ne[2],
                                  (int32_t)t->ne[3]};
                int n_dims = 1;
                for (int d = 1; d < 4; d++)
                    if (t->ne[d] > 1)
                        n_dims = d + 1;
                hdr[1] = n_dims;
                fwrite(hdr, sizeof(int32_t), 6, f);
                fwrite(buf.data(), 1, nb, f);
                fclose(f);
            }
        };
        dump("audio_subsample_output");
        for (int il = 0; il < (int)ahp.num_layers; il++) {
            char nm[32];
            snprintf(nm, sizeof(nm), "audio_layer_%d", il);
            dump(nm);
        }
        dump("audio_tower_output");
    }

    ggml_free(ectx);

    *out_T_enc = N_audio;
    *out_d_model = proj_dim;
    return out;
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
