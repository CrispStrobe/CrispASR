// cohere.cpp — Cohere Transcribe inference via ggml
//
// Architecture:
//   Encoder: Conv2D subsampling (×8) + 48-layer Conformer (Transformer-XL rel-pos attention)
//   Decoder: 8-layer causal transformer with cross-attention + KV cache
//   Features: on-the-fly preemphasis → STFT → mel filterbank → log → per-feature norm
//
// Tensor naming follows export_gguf.py / cohere-arch.h.

#include "cohere.h"
#include "cohere-arch.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdarg>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fftw3.h>
#include <cblas.h>
#if defined(__F16C__) && defined(__AVX2__)
#include <immintrin.h>
#define CT_HAVE_F16C 1
#endif

// ---------------------------------------------------------------------------
// Logging & Debugging
// ---------------------------------------------------------------------------

static bool cohere_debug_enabled(void) {
    static bool init = false;
    static bool enabled = false;
    if (!init) { enabled = getenv("COHERE_DEBUG") != nullptr; init = true; }
    return enabled;
}

static bool cohere_bench_enabled(void) {
    static bool init = false;
    static bool enabled = false;
    if (!init) { enabled = getenv("COHERE_BENCH") != nullptr; init = true; }
    return enabled;
}

static void cohere_debug(const char * fmt, ...) {
    if (!cohere_debug_enabled()) return;
    va_list args; va_start(args, fmt); vprintf(fmt, args); va_end(args);
    fflush(stdout);
}

static void cohere_log_tensor(const char * name, const struct ggml_tensor * t) {
    if (!cohere_debug_enabled()) return;
    if (!t) { cohere_debug("%-25s: NULL\n", name); return; }
    cohere_debug("%-25s: shape [%4ld, %4ld, %4ld, %4ld] type %d\n",
        name,
        (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3],
        (int)t->type);
}

// ---------------------------------------------------------------------------
// Performance counters — accumulated per cohere_transcribe() call
// ---------------------------------------------------------------------------

struct cohere_perf {
    int64_t t_features_us    = 0; // STFT + mel filterbank
    int64_t t_enc_build_us   = 0; // encoder graph construction
    int64_t t_enc_alloc_us   = 0; // encoder ggml_backend_sched_alloc_graph
    int64_t t_enc_compute_us = 0; // encoder ggml_backend_sched_graph_compute
    int64_t t_cross_kv_us    = 0; // copying cross-KV tensors from encoder output
    int64_t t_dec_build_us   = 0; // decoder graph build (all steps summed)
    int64_t t_dec_alloc_us   = 0; // decoder sched alloc (all steps summed)
    int64_t t_dec_compute_us = 0; // decoder compute (all steps summed)
    int64_t t_dec_logits_us  = 0; // decoder ggml_backend_tensor_get logits (all steps)
    int64_t t_dec_step_min_us = INT64_MAX;
    int64_t t_dec_step_max_us = 0;
    int     n_dec_steps       = 0; // total autoregressive steps (prompt + generated)
    int64_t t_total_us        = 0; // wall time for entire cohere_transcribe()
    // Graph sizes
    int enc_n_nodes           = 0;
    int dec_n_nodes_prompt    = 0; // first decode call (full prompt batch)
    int dec_n_nodes_step      = 0; // subsequent calls (n_tok=1)
    // Memory (bytes, snapshot after init / after cross-kv alloc)
    size_t mem_model_buf      = 0;
    size_t mem_kv_buf         = 0;
    size_t mem_cross_kv_buf   = 0;
    size_t mem_sched_buf      = 0;
    size_t mem_compute_meta   = 0;
};

static void cohere_perf_reset(cohere_perf & p) {
    p = cohere_perf{};
}

static void cohere_perf_print(const cohere_perf & p, int n_samples, int sample_rate) {
    double audio_sec = (double)n_samples / sample_rate;
    double total_sec = p.t_total_us / 1e6;
    double rtf       = (total_sec > 0) ? (audio_sec / total_sec) : 0.0;

    fprintf(stderr, "\n");
    fprintf(stderr, "cohere: =========== performance report ===========\n");
    fprintf(stderr, "cohere:  audio            %.2f s\n", audio_sec);
    fprintf(stderr, "cohere:  total wall       %7.1f ms   (%.3f× real-time)\n",
            p.t_total_us / 1e3, rtf);
    fprintf(stderr, "cohere: ----- feature extraction -----\n");
    fprintf(stderr, "cohere:  features         %7.1f ms\n", p.t_features_us / 1e3);
    fprintf(stderr, "cohere: ----- encoder -----\n");
    fprintf(stderr, "cohere:  enc graph build  %7.1f ms   nodes=%d\n",
            p.t_enc_build_us / 1e3, p.enc_n_nodes);
    fprintf(stderr, "cohere:  enc sched alloc  %7.1f ms\n", p.t_enc_alloc_us / 1e3);
    fprintf(stderr, "cohere:  enc compute      %7.1f ms\n", p.t_enc_compute_us / 1e3);
    fprintf(stderr, "cohere:  cross-kv copy    %7.1f ms\n", p.t_cross_kv_us / 1e3);
    fprintf(stderr, "cohere:  enc total        %7.1f ms\n",
            (p.t_enc_build_us + p.t_enc_alloc_us + p.t_enc_compute_us + p.t_cross_kv_us) / 1e3);
    fprintf(stderr, "cohere: ----- decoder (%d steps) -----\n", p.n_dec_steps);
    if (p.n_dec_steps > 0) {
        double dprompt = p.dec_n_nodes_prompt;
        double dstep   = p.dec_n_nodes_step;
        fprintf(stderr, "cohere:  dec graph build  %7.1f ms   %.2f ms/step  (nodes: prompt=%d step=%d)\n",
                p.t_dec_build_us / 1e3,
                p.t_dec_build_us / 1e3 / p.n_dec_steps,
                p.dec_n_nodes_prompt, p.dec_n_nodes_step);
        (void)dprompt; (void)dstep;
        fprintf(stderr, "cohere:  dec sched alloc  %7.1f ms   %.2f ms/step\n",
                p.t_dec_alloc_us / 1e3,
                p.t_dec_alloc_us / 1e3 / p.n_dec_steps);
        fprintf(stderr, "cohere:  dec compute      %7.1f ms   %.2f ms/step  min=%.2f max=%.2f\n",
                p.t_dec_compute_us / 1e3,
                p.t_dec_compute_us / 1e3 / p.n_dec_steps,
                (p.t_dec_step_min_us == INT64_MAX ? 0 : p.t_dec_step_min_us) / 1e3,
                p.t_dec_step_max_us / 1e3);
        fprintf(stderr, "cohere:  dec logits get   %7.1f ms   %.2f ms/step\n",
                p.t_dec_logits_us / 1e3,
                p.t_dec_logits_us / 1e3 / p.n_dec_steps);
        fprintf(stderr, "cohere:  dec total        %7.1f ms\n",
                (p.t_dec_build_us + p.t_dec_alloc_us + p.t_dec_compute_us + p.t_dec_logits_us) / 1e3);
    }
    fprintf(stderr, "cohere: ----- memory -----\n");
    fprintf(stderr, "cohere:  model weights    %7.1f MiB\n", p.mem_model_buf  / 1048576.0);
    fprintf(stderr, "cohere:  kv cache         %7.1f MiB\n", p.mem_kv_buf     / 1048576.0);
    fprintf(stderr, "cohere:  cross-kv cache   %7.1f MiB\n", p.mem_cross_kv_buf / 1048576.0);
    fprintf(stderr, "cohere:  sched buf        %7.1f MiB\n", p.mem_sched_buf  / 1048576.0);
    fprintf(stderr, "cohere:  compute_meta     %7.1f KiB\n", p.mem_compute_meta / 1024.0);
    fprintf(stderr, "cohere: ================================================\n\n");
}

// ---------------------------------------------------------------------------
// Per-op profiling (COHERE_PROF=1)
// ---------------------------------------------------------------------------

struct cohere_op_prof {
    int64_t t_us  = 0;
    int     count = 0;
};

struct cohere_prof_state {
    // coarse buckets
    cohere_op_prof mul_mat;    // GGML_OP_MUL_MAT
    cohere_op_prof soft_max;   // GGML_OP_SOFT_MAX
    cohere_op_prof norm;       // GGML_OP_NORM / RMS_NORM
    cohere_op_prof cont;       // GGML_OP_CONT
    cohere_op_prof im2col;     // GGML_OP_IM2COL (part of conv1d_dw expansion)
    cohere_op_prof add;        // GGML_OP_ADD / GGML_OP_MUL / GGML_OP_SCALE
    cohere_op_prof other;      // everything else (view, reshape, permute, silu, etc.)
    int64_t t_node_start = 0;  // wall time before node execute (ask=true)
};

static bool cohere_prof_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ps = (cohere_prof_state *)user_data;
    if (ask) {
        ps->t_node_start = ggml_time_us();
        return true;
    }
    int64_t dt = ggml_time_us() - ps->t_node_start;
    switch (t->op) {
        case GGML_OP_MUL_MAT:     ps->mul_mat.t_us  += dt; ps->mul_mat.count++;  break;
        case GGML_OP_SOFT_MAX:    ps->soft_max.t_us += dt; ps->soft_max.count++; break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:    ps->norm.t_us     += dt; ps->norm.count++;     break;
        case GGML_OP_CONT:        ps->cont.t_us     += dt; ps->cont.count++;     break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:   ps->im2col.t_us   += dt; ps->im2col.count++;  break;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:       ps->add.t_us      += dt; ps->add.count++;     break;
        default:                  ps->other.t_us    += dt; ps->other.count++;    break;
    }
    return true;
}

static void cohere_prof_print(const cohere_prof_state & ps) {
    auto pct = [&](int64_t v, int64_t total) {
        return total > 0 ? 100.0 * v / total : 0.0;
    };
    int64_t total = ps.mul_mat.t_us + ps.soft_max.t_us + ps.norm.t_us
                  + ps.cont.t_us + ps.im2col.t_us + ps.add.t_us + ps.other.t_us;
    fprintf(stderr, "cohere: -------- encoder op profile (COHERE_PROF) --------\n");
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n",
            "mul_mat",   ps.mul_mat.t_us/1e3,   pct(ps.mul_mat.t_us,total),   ps.mul_mat.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n",
            "soft_max",  ps.soft_max.t_us/1e3,  pct(ps.soft_max.t_us,total),  ps.soft_max.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n",
            "norm",      ps.norm.t_us/1e3,      pct(ps.norm.t_us,total),      ps.norm.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n",
            "cont",      ps.cont.t_us/1e3,      pct(ps.cont.t_us,total),      ps.cont.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n",
            "im2col",    ps.im2col.t_us/1e3,    pct(ps.im2col.t_us,total),    ps.im2col.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n",
            "add/mul/sc", ps.add.t_us/1e3,      pct(ps.add.t_us,total),       ps.add.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n",
            "other",     ps.other.t_us/1e3,     pct(ps.other.t_us,total),     ps.other.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  (measured sum; real enc may differ due to overhead)\n",
            "TOTAL",     total/1e3);
    fprintf(stderr, "cohere: -------------------------------------------------------\n");
}

// ---------------------------------------------------------------------------
// Helpers

// Like whisper.cpp's ggml_graph_compute_helper: set n_threads on every backend
// in the scheduler (via registry proc address) before each compute call.
// This ensures the thread count is applied correctly even after sched resets.
static bool cohere_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * gf, int n_threads) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); i++) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto * fn = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn) fn(backend, n_threads);
        }
    }
    return ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Model hyperparams
// ---------------------------------------------------------------------------

struct cohere_hparams {
    int vocab_size   = 16384;
    // encoder
    int enc_n_layers = 48;
    int enc_d_model  = 1280;
    int enc_n_heads  = 8;
    int enc_head_dim = 160;
    int enc_ffn_dim  = 5120;
    int enc_conv_k   = 9;
    // decoder
    int dec_n_layers = 8;
    int dec_d_model  = 1024;
    int dec_n_heads  = 8;
    int dec_head_dim = 128;
    int dec_ffn_dim  = 4096;
    int dec_max_ctx  = 1024;
    // audio
    int sample_rate  = 16000;
    int n_mels       = 128;
    int n_fft        = 512;
    int hop_length   = 160;
    int win_length   = 400;
    // derived
    int n_freqs() const { return n_fft / 2 + 1; }            // 257
    int pre_conv_ch  = 256;
    int pre_sub_fac  = 8;  // 3 × stride-2 → ×8 downsampling
};

// ---------------------------------------------------------------------------
// Conformer layer weights
// ---------------------------------------------------------------------------

struct cohere_enc_layer {
    // FF1
    ggml_tensor * ff1_norm_w, * ff1_norm_b;
    ggml_tensor * ff1_up_w, * ff1_up_b;
    ggml_tensor * ff1_dn_w, * ff1_dn_b;
    // Self-attention (relative pos)
    ggml_tensor * attn_norm_w, * attn_norm_b;
    ggml_tensor * attn_q_w,    * attn_q_b;
    ggml_tensor * attn_k_w,    * attn_k_b;
    ggml_tensor * attn_v_w,    * attn_v_b;
    ggml_tensor * attn_qkv_w,  * attn_qkv_b; // combined
    ggml_tensor * attn_out_w,  * attn_out_b;

    ggml_tensor * attn_pos_w;          // linear_pos  [d,d]
    ggml_tensor * attn_pos_bias_u;     // [heads, head_dim]
    ggml_tensor * attn_pos_bias_v;
    // Convolution module
    ggml_tensor * conv_norm_w, * conv_norm_b;
    ggml_tensor * conv_pw1_w, * conv_pw1_b; // pointwise1: [2d, d, 1]
    ggml_tensor * conv_dw_w,  * conv_dw_b;  // depthwise:  [d, 1, k]
    ggml_tensor * conv_bn_w,  * conv_bn_b;  // batch-norm scale/bias
    ggml_tensor * conv_bn_mean, * conv_bn_var;
    ggml_tensor * conv_pw2_w, * conv_pw2_b; // pointwise2: [d, d, 1]
    // FF2
    ggml_tensor * ff2_norm_w, * ff2_norm_b;
    ggml_tensor * ff2_up_w, * ff2_up_b;
    ggml_tensor * ff2_dn_w, * ff2_dn_b;
    // Output norm
    ggml_tensor * out_norm_w, * out_norm_b;

    // BatchNorm-folded depthwise conv weights (precomputed at model load time).
    // w_fused[d] = dw_w[d] * bn_gamma[d] / sqrt(bn_var[d] + 1e-5)
    // b_fused[d] = (dw_b[d] - bn_mean[d]) * bn_gamma[d] / sqrt(bn_var[d] + 1e-5) + bn_beta[d]
    // These are F32 host vectors used by both the legacy and ggml-graph paths.
    std::vector<float> conv_dw_w_fused; // [d * conv_k]
    std::vector<float> conv_dw_b_fused; // [d]
};

// ---------------------------------------------------------------------------
// Decoder layer weights
// ---------------------------------------------------------------------------

struct cohere_dec_layer {
    ggml_tensor * attn_ln_w, * attn_ln_b;
    ggml_tensor * attn_q_w,  * attn_q_b;
    ggml_tensor * attn_k_w,  * attn_k_b;
    ggml_tensor * attn_v_w,  * attn_v_b;
    ggml_tensor * attn_o_w,  * attn_o_b;
    ggml_tensor * cross_ln_w, * cross_ln_b;
    ggml_tensor * cross_q_w,  * cross_q_b;
    ggml_tensor * cross_k_w,  * cross_k_b;
    ggml_tensor * cross_v_w,  * cross_v_b;
    ggml_tensor * cross_o_w,  * cross_o_b;
    ggml_tensor * ffn_ln_w,  * ffn_ln_b;
    ggml_tensor * ffn_up_w,  * ffn_up_b;
    ggml_tensor * ffn_dn_w,  * ffn_dn_b;
};

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

struct cohere_model {
    cohere_hparams hparams;

    // Feature extraction
    ggml_tensor * fe_mel_fb; // [1, n_mels, n_freqs]
    ggml_tensor * fe_window; // [win_length]

    // Pre-encode subsampling
    ggml_tensor * pre_conv0_w, * pre_conv0_b;
    ggml_tensor * pre_conv2_w, * pre_conv2_b;
    ggml_tensor * pre_conv3_w, * pre_conv3_b;
    ggml_tensor * pre_conv5_w, * pre_conv5_b;
    ggml_tensor * pre_conv6_w, * pre_conv6_b;
    ggml_tensor * pre_out_w,   * pre_out_b;

    // Encoder layers
    std::vector<cohere_enc_layer> enc_layers;

    // Encoder→decoder projection
    ggml_tensor * enc_proj_w, * enc_proj_b;

    // Decoder
    ggml_tensor * dec_emb_w;
    ggml_tensor * dec_pos_w;
    ggml_tensor * dec_emb_ln_w, * dec_emb_ln_b;
    std::vector<cohere_dec_layer> dec_layers;
    ggml_tensor * dec_out_ln_w, * dec_out_ln_b;
    ggml_tensor * dec_head_w,   * dec_head_b;

    // ggml bookkeeping
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
};

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

struct cohere_vocab {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;

    int n_vocab() const { return (int)id_to_token.size(); }

    int token_id(const std::string & s) const {
        auto it = token_to_id.find(s);
        return it == token_to_id.end() ? -1 : it->second;
    }
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct cohere_context {
    cohere_model  model;
    cohere_vocab  vocab;
    cohere_context_params params;

    // Persistent KV cache tensors and their context
    struct ggml_context * kv_ctx  = nullptr;
    struct ggml_tensor  * kv_k    = nullptr;
    struct ggml_tensor  * kv_v    = nullptr;
    struct ggml_tensor  * kq_mask = nullptr; // [max_ctx] F32, additive mask for fixed-shape attention

    // Cross-attention KV cache: computed once per utterance from encoder output.
    // Shape per layer: (dec_d_model, T_enc)
    struct ggml_context * cross_kv_ctx = nullptr;
    struct ggml_backend_buffer * cross_kv_buf = nullptr;
    std::vector<struct ggml_tensor *> cross_kv_k;
    std::vector<struct ggml_tensor *> cross_kv_v;

    // ggml backends: primary (GPU if available, else CPU) + CPU fallback for unsupported ops
    ggml_backend_t       ggml_backend     = nullptr; // primary backend (Metal/CUDA/CPU)
    ggml_backend_t       ggml_backend_cpu = nullptr; // always CPU; used as sched fallback
    ggml_backend_sched_t ggml_alloc       = nullptr;

    // Metadata context for graph node descriptors (no_alloc=true; actual buffers via gallocr)
    // Sized generously; only holds ggml_tensor_overhead() * N_NODES bytes.
    std::vector<uint8_t> compute_meta;

    // Cached T_enc from last encode call, needed by decode graph builder.
    int cached_T_enc = 0;

    // Mel spectrogram buffer
    std::vector<float> mel_buf;

    // Constants
    struct ggml_context * ctx_const = nullptr;
    struct ggml_tensor * eps = nullptr;

    // Cached decoder graph for n_tokens=1
    struct ggml_cgraph * gf_decode_1 = nullptr;
    int cached_offset_1 = -1;

    // Performance counters (reset at start of each cohere_transcribe())
    cohere_perf perf;
};

static void cohere_log_tensor(const char * name, const struct ggml_tensor * t);
static struct ggml_cgraph * cohere_build_graph_encoder(struct cohere_context * ctx, int T_mel);
static struct ggml_cgraph * cohere_build_graph_decoder(struct cohere_context * ctx, const int * tokens, int n_tokens, int offset);
static void cohere_fold_batchnorm(cohere_model & model);

// ---------------------------------------------------------------------------
// Encoder Graph Builder
// ---------------------------------------------------------------------------

static struct ggml_tensor * cohere_rel_shift(struct ggml_context * ctx, struct ggml_tensor * a) {
    const int T = a->ne[1];
    const int n_heads = a->ne[2];
    struct ggml_tensor * out = ggml_view_3d(ctx, a,
        T, T, n_heads,
        a->nb[1] - a->nb[0],
        a->nb[2],
        (T - 1) * a->nb[0]
    );
    return out;
}

static struct ggml_cgraph * cohere_build_graph_encoder(struct cohere_context * ctx, int T_mel) {
    const auto & model = ctx->model;
    const auto & hp    = model.hparams;
    const int d        = hp.enc_d_model;
    const int n_heads  = hp.enc_n_heads;
    const int head_dim = hp.enc_head_dim;
    const int n_mels   = hp.n_mels;

    struct ggml_init_params params = {
        .mem_size   = ctx->compute_meta.size(),
        .mem_buffer = ctx->compute_meta.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);

    // mel: [n_mels, T_mel]
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // --- Subsampling ---
    // conv.0: [n_mels, T_mel, 1] -> [64, T/2, 256]
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, model.pre_conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv0_b, 1, 1, model.pre_conv0_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // conv.2: depthwise [64, T/2, 256] -> [32, T/4, 256]
    cur = ggml_conv_2d_dw(ctx0, model.pre_conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv2_b, 1, 1, model.pre_conv2_b->ne[0], 1), GGML_TYPE_F32));
    // conv.3: pointwise [32, T/4, 256] -> [32, T/4, 256]
    cur = ggml_conv_2d(ctx0, model.pre_conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv3_b, 1, 1, model.pre_conv3_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // conv.5: depthwise [32, T/4, 256] -> [16, T/8, 256]
    cur = ggml_conv_2d_dw(ctx0, model.pre_conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv5_b, 1, 1, model.pre_conv5_b->ne[0], 1), GGML_TYPE_F32));
    // conv.6: pointwise [16, T/8, 256] -> [16, T/8, 256]
    cur = ggml_conv_2d(ctx0, model.pre_conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv6_b, 1, 1, model.pre_conv6_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // Flatten: [W3, H3, 256] -> [W3, 256, H3] -> [W3*256, H3]
    int H3 = cur->ne[1];
    int W3 = cur->ne[0];
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3)); // [W3, C, H3]
    cur = ggml_reshape_2d(ctx0, cur, W3 * 256, H3);
    cohere_log_tensor("after_flatten", cur);

    // pre_out: [4096, H3] -> [d, H3]
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.pre_out_w, cur), model.pre_out_b);

    const int T = H3;

    // Positional encodings sinusoidal: [d, 2*T-1]
    struct ggml_tensor * pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto & layer = model.enc_layers[il];
        struct ggml_tensor * inpL = cur;

        // FF1
        struct ggml_tensor * ff1 = ggml_norm(ctx0, cur, 1e-5f);
        ff1 = ggml_mul_inplace(ctx0, ff1, layer.ff1_norm_w);
        ff1 = ggml_add_inplace(ctx0, ff1, layer.ff1_norm_b);
        ff1 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff1_up_w, ff1), layer.ff1_up_b);
        ff1 = ggml_silu_inplace(ctx0, ff1);
        ff1 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff1_dn_w, ff1), layer.ff1_dn_b);
        cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, ff1, 0.5f));

        struct ggml_tensor * inpAtn = cur;

        // Self-Attention
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.attn_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.attn_norm_b);
        
        struct ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_q_w, cur), layer.attn_q_b);
        struct ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_k_w, cur), layer.attn_k_b);
        struct ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_v_w, cur), layer.attn_v_b);
        
        struct ggml_tensor * R = ggml_mul_mat(ctx0, layer.attn_pos_w, pos_enc); // [d, 2T-1]
        
        struct ggml_tensor * Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_u, d));
        struct ggml_tensor * Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_v, d));
        
        Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3); // [hd, T, H]
        Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3); // [hd, T, H]
        K   = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K,   head_dim, n_heads, T), 0, 2, 1, 3); // [hd, T, H]
        R   = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R,   head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3); // [hd, 2T-1, H]
        
        struct ggml_tensor * AC = ggml_mul_mat(ctx0, ggml_cont(ctx0, K), ggml_cont(ctx0, Q_u)); // [T, T, H]
        struct ggml_tensor * BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), ggml_cont(ctx0, Q_v)); // [2T-1, T, H]

        if (il == 0) {
            cohere_log_tensor("K", K);
            cohere_log_tensor("Q_u", Q_u);
            cohere_log_tensor("AC", AC);
        }

        struct ggml_tensor * BD = cohere_rel_shift(ctx0, BD_raw); // [T, T, H]
        
        struct ggml_tensor * scores = ggml_add_inplace(ctx0, AC, BD);
        scores = ggml_scale_inplace(ctx0, scores, 1.0f / sqrtf((float)head_dim));
        scores = ggml_soft_max_inplace(ctx0, scores);

        struct ggml_tensor * V_reshaped = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T);
        if (il == 0) cohere_log_tensor("V_reshaped", V_reshaped);

        struct ggml_tensor * V_trans = ggml_permute(ctx0, V_reshaped, 1, 2, 0, 3); // [T, hd, H]
        struct ggml_tensor * attn_out = ggml_mul_mat(ctx0, ggml_cont(ctx0, V_trans), scores); // [hd, T, H]

        if (il == 0) {
            cohere_log_tensor("scores", scores);
            cohere_log_tensor("V_trans", V_trans);
            cohere_log_tensor("attn_out", attn_out);
        }

        attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);

        attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_out_w, attn_out), layer.attn_out_b);
        cur = ggml_add(ctx0, inpAtn, attn_out);

        struct ggml_tensor * inpCnv = cur;

        // Convolution module
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.conv_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.conv_norm_b);
        
        struct ggml_tensor * pw1_w = ggml_reshape_2d(ctx0, layer.conv_pw1_w, d, 2 * d);
        struct ggml_tensor * cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw1_w, cur), layer.conv_pw1_b);

        struct ggml_tensor * cnv_gate = ggml_view_2d(ctx0, cnv, d, T, cnv->nb[1], d * sizeof(float));
        cnv = ggml_mul(ctx0, 
            ggml_view_2d(ctx0, cnv, d, T, cnv->nb[1], 0),
            ggml_sigmoid(ctx0, cnv_gate)
        );
        
        // cnv is [C, T] initially.
        // ggml_conv_1d_dw handles [T, C] input (ne0=T, ne1=C).
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv)); // [C, T] -> [T, C]
        cnv = ggml_conv_1d_dw(ctx0, layer.conv_dw_w, cnv, 1, (hp.enc_conv_k - 1) / 2, 1);
        
        // ggml_conv_1d_dw returns [T, 1, C, 1] (ne0=T, ne1=1, ne2=C)
        // Permute to [C, T, 1, 1] to match the rest of the layer expectation.
        // p0 is where old_ne0 (T) goes -> 1
        // p1 is where old_ne1 (1) goes -> 2
        // p2 is where old_ne2 (C) goes -> 0
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
        
        // Add folded bias (BN folded into conv_dw_w/b at init by cohere_fold_batchnorm)
        cnv = ggml_add(ctx0, cnv, ggml_reshape_4d(ctx0, layer.conv_dw_b, d, 1, 1, 1));
        
        cnv = ggml_silu_inplace(ctx0, cnv); // swish
        
        struct ggml_tensor * pw2_w = ggml_reshape_2d(ctx0, layer.conv_pw2_w, d, d);
        cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw2_w, cnv), layer.conv_pw2_b);
        cur = ggml_add(ctx0, inpCnv, cnv);

        struct ggml_tensor * inpFF2 = cur;

        // FF2
        struct ggml_tensor * ff2 = ggml_norm(ctx0, cur, 1e-5f);
        ff2 = ggml_mul_inplace(ctx0, ff2, layer.ff2_norm_w);
        ff2 = ggml_add_inplace(ctx0, ff2, layer.ff2_norm_b);
        ff2 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff2_up_w, ff2), layer.ff2_up_b);
        ff2 = ggml_silu_inplace(ctx0, ff2);
        ff2 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff2_dn_w, ff2), layer.ff2_dn_b);
        cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, ff2, 0.5f));

        // Final output norm for the layer
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.out_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.out_norm_b);
    }

    // Encoder-decoder projection
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.enc_proj_w, cur), model.enc_proj_b);
    ggml_set_name(cur, "enc_out");

    // Pre-compute cross KV for all decoder layers
    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto & layer = model.dec_layers[il];
        
        struct ggml_tensor * ck = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_k_w, cur), layer.cross_k_b);
        struct ggml_tensor * cv = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_v_w, cur), layer.cross_v_b);
        
        // CK: [d, T] -> [hd, H, T] -> [hd, T, H]
        struct ggml_tensor * ck_ready = ggml_cont(ctx0, ggml_permute(ctx0, 
            ggml_reshape_3d(ctx0, ck, hp.dec_head_dim, hp.dec_n_heads, T), 0, 2, 1, 3));
        
        // CV: [d, T] -> [hd, H, T] -> [T, hd, H] (pre-transposed for decoder mul_mat)
        struct ggml_tensor * cv_ready = ggml_cont(ctx0, ggml_permute(ctx0, 
            ggml_reshape_3d(ctx0, cv, hp.dec_head_dim, hp.dec_n_heads, T), 1, 2, 0, 3));

        char ck_name[32], cv_name[32];
        snprintf(ck_name, sizeof(ck_name), "ck_%d", il);
        snprintf(cv_name, sizeof(cv_name), "cv_%d", il);
        ggml_set_name(ck_ready, ck_name);
        ggml_set_name(cv_ready, cv_name);
        
        ggml_build_forward_expand(gf, ck_ready);
        ggml_build_forward_expand(gf, cv_ready);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}


static struct ggml_cgraph * cohere_build_graph_decoder(struct cohere_context * ctx, const int * tokens, int n_tokens, int offset);

#define CT_CHECK(x) do { if (!(x)) { fprintf(stderr, "CT_CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); abort(); } } while(0)

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float swish(float x)   { return x * sigmoid(x); }

// ---------------------------------------------------------------------------
// GGUF loading helpers
// ---------------------------------------------------------------------------

#include "gguf.h"

static ggml_tensor * ct_get_tensor(cohere_model & model, const std::string & name) {
    auto it = model.tensors.find(name);
    if (it == model.tensors.end()) {
        fprintf(stderr, "cohere: tensor '%s' not found in GGUF\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

static ggml_tensor * ct_get_tensor_fmt(cohere_model & model, const char * fmt, int idx) {
    char buf[128];
    snprintf(buf, sizeof(buf), fmt, idx);
    return ct_get_tensor(model, buf);
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static bool cohere_load_model(cohere_model & model,
                               cohere_vocab  & vocab,
                               const char * path,
                               ggml_backend_t backend) {
    // First pass: read metadata
    struct ggml_init_params meta_params = {
        .mem_size   = 4 * 1024 * 1024, // 4 MB for metadata
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * meta_ctx = ggml_init(meta_params);
    struct gguf_init_params load_params_meta = { .no_alloc = true, .ctx = &meta_ctx };
    struct gguf_context * gguf_ctx = gguf_init_from_file(path, load_params_meta);
    if (!gguf_ctx) {
        fprintf(stderr, "cohere: failed to open '%s'\n", path);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    auto & hp = model.hparams;
    auto kv_i = [&](const char * key) -> int {
        int ki = gguf_find_key(gguf_ctx, key);
        if (ki < 0) { fprintf(stderr, "cohere: missing key '%s'\n", key); return 0; }
        return (int)gguf_get_val_u32(gguf_ctx, ki);
    };

    hp.vocab_size   = kv_i(CT_KEY_VOCAB_SIZE);
    hp.enc_n_layers = kv_i(CT_KEY_ENC_N_LAYERS);
    hp.enc_d_model  = kv_i(CT_KEY_ENC_D_MODEL);
    hp.enc_n_heads  = kv_i(CT_KEY_ENC_N_HEADS);
    hp.enc_head_dim = kv_i(CT_KEY_ENC_HEAD_DIM);
    hp.enc_ffn_dim  = kv_i(CT_KEY_ENC_FFN_DIM);
    hp.enc_conv_k   = kv_i(CT_KEY_ENC_CONV_KERNEL);
    hp.dec_n_layers = kv_i(CT_KEY_DEC_N_LAYERS);
    hp.dec_d_model  = kv_i(CT_KEY_DEC_D_MODEL);
    hp.dec_n_heads = kv_i(CT_KEY_DEC_N_HEADS);
    hp.dec_head_dim = kv_i(CT_KEY_DEC_HEAD_DIM);
    hp.dec_ffn_dim  = kv_i(CT_KEY_DEC_FFN_DIM);
    hp.dec_max_ctx  = kv_i(CT_KEY_DEC_MAX_CTX);
    hp.n_mels       = kv_i(CT_KEY_AUDIO_N_MELS);
    hp.n_fft        = kv_i(CT_KEY_AUDIO_N_FFT);
    hp.hop_length   = kv_i(CT_KEY_AUDIO_HOP);
    hp.win_length   = kv_i(CT_KEY_AUDIO_WIN);

    // Load vocabulary
    {
        int ki = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
        if (ki >= 0) {
            int n = gguf_get_arr_n(gguf_ctx, ki);
            cohere_debug("cohere: loading %d tokens from GGUF\n", n);
            vocab.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                vocab.id_to_token[i] = gguf_get_arr_str(gguf_ctx, ki, i);
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
            const char * specials[] = {"<|startoftranscript|>", "<|en|>", "<|endoftext|>"};
            for (const char * s : specials) {
                cohere_debug("cohere: token '%s' -> ID %d\n", s, vocab.token_id(s));
            }
        } else {
            cohere_debug("cohere: WARNING: tokenizer.ggml.tokens not found in GGUF\n");
        }
    }

    gguf_free(gguf_ctx);
    ggml_free(meta_ctx);

    // Second pass: load all tensor metadata (no_alloc=true)
    struct ggml_context * weight_ctx = nullptr;
    {
        struct gguf_init_params load_params = { .no_alloc = true, .ctx = &weight_ctx };
        gguf_ctx = gguf_init_from_file(path, load_params);
        if (!gguf_ctx || !weight_ctx) {
            fprintf(stderr, "cohere: failed to load tensor metadata from '%s'\n", path);
            return false;
        }

        // Create backend buffer for all tensors
        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        // mmap the GGUF file — one syscall instead of N fread calls, zero per-tensor heap alloc
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "cohere: failed to open '%s' for mmap\n", path);
            return false;
        }
        struct stat st;
        fstat(fd, &st);
        size_t file_size = (size_t)st.st_size;
        void * mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmap_base == MAP_FAILED) {
            fprintf(stderr, "cohere: mmap failed for '%s', falling back to fread\n", path);
            mmap_base = nullptr;
        }

        size_t data_offset = gguf_get_data_offset(gguf_ctx);

        for (ggml_tensor * t = ggml_get_first_tensor(weight_ctx); t;
             t = ggml_get_next_tensor(weight_ctx, t)) {
            model.tensors[ggml_get_name(t)] = t;

            int64_t tensor_id = gguf_find_tensor(gguf_ctx, ggml_get_name(t));
            if (tensor_id < 0) continue;

            size_t t_offset = gguf_get_tensor_offset(gguf_ctx, tensor_id);
            size_t nbytes   = ggml_nbytes(t);

            if (mmap_base) {
                // Direct pointer into mmap region — no heap allocation, no fread
                ggml_backend_tensor_set(t, (const char *)mmap_base + data_offset + t_offset, 0, nbytes);
            } else {
                // Fallback: fread per tensor
                FILE * file = fopen(path, "rb");
                if (file) {
                    fseek(file, (long)(data_offset + t_offset), SEEK_SET);
                    std::vector<uint8_t> tmp(nbytes);
                    if (fread(tmp.data(), 1, nbytes, file) == nbytes)
                        ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
                    fclose(file);
                }
            }
        }

        if (mmap_base) munmap(mmap_base, file_size);

        model.ctx = weight_ctx;
        gguf_free(gguf_ctx);

        struct ggml_tensor * tp = model.tensors["enc.proj.weight"];
        if (tp) {
            std::vector<uint8_t> data(ggml_nbytes(tp));
            ggml_backend_tensor_get(tp, data.data(), 0, data.size());
            if (tp->type == GGML_TYPE_F16) {
                float v0 = ggml_fp16_to_fp32(((ggml_fp16_t*)data.data())[0]);
                cohere_debug("cohere: enc.proj.weight [0] (F16): %.4f\n", v0);
            } else if (tp->type == GGML_TYPE_F32) {
                cohere_debug("cohere: enc.proj.weight [0] (F32): %.4f\n", ((float*)data.data())[0]);
            }
        }
    }

    // Wire up model fields
    auto & m = model;
    auto T = [&](const char * name) { return ct_get_tensor(m, name); };
    auto TF = [&](const char * fmt, int i) { return ct_get_tensor_fmt(m, fmt, i); };

    m.fe_mel_fb = T(CT_FE_MEL_FB);
    m.fe_window = T(CT_FE_WINDOW);

    m.pre_conv0_w = T(CT_PRE_CONV0_W);  m.pre_conv0_b = T(CT_PRE_CONV0_B);
    m.pre_conv2_w = T(CT_PRE_CONV2_W);  m.pre_conv2_b = T(CT_PRE_CONV2_B);
    m.pre_conv3_w = T(CT_PRE_CONV3_W);  m.pre_conv3_b = T(CT_PRE_CONV3_B);
    m.pre_conv5_w = T(CT_PRE_CONV5_W);  m.pre_conv5_b = T(CT_PRE_CONV5_B);
    m.pre_conv6_w = T(CT_PRE_CONV6_W);  m.pre_conv6_b = T(CT_PRE_CONV6_B);
    m.pre_out_w   = T(CT_PRE_OUT_W);    m.pre_out_b   = T(CT_PRE_OUT_B);

    m.enc_layers.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        auto & l = m.enc_layers[i];
        l.ff1_norm_w  = TF(CT_ENC_FF1_NORM_W, i); l.ff1_norm_b  = TF(CT_ENC_FF1_NORM_B, i);
        l.ff1_up_w    = TF(CT_ENC_FF1_UP_W,   i); l.ff1_up_b    = TF(CT_ENC_FF1_UP_B,   i);
        l.ff1_dn_w    = TF(CT_ENC_FF1_DN_W,   i); l.ff1_dn_b    = TF(CT_ENC_FF1_DN_B,   i);
        l.attn_norm_w = TF(CT_ENC_ATN_NORM_W, i); l.attn_norm_b = TF(CT_ENC_ATN_NORM_B, i);
        l.attn_q_w    = TF(CT_ENC_ATN_Q_W,    i); l.attn_q_b    = TF(CT_ENC_ATN_Q_B,    i);
        l.attn_k_w    = TF(CT_ENC_ATN_K_W,    i); l.attn_k_b    = TF(CT_ENC_ATN_K_B,    i);
        l.attn_v_w    = TF(CT_ENC_ATN_V_W,    i); l.attn_v_b    = TF(CT_ENC_ATN_V_B,    i);
        l.attn_out_w  = TF(CT_ENC_ATN_OUT_W,  i); l.attn_out_b  = TF(CT_ENC_ATN_OUT_B,  i);
        l.attn_pos_w  = TF(CT_ENC_ATN_POS_W,  i);
        l.attn_pos_bias_u = TF(CT_ENC_ATN_POS_U, i);
        l.attn_pos_bias_v = TF(CT_ENC_ATN_POS_V, i);
        l.conv_norm_w = TF(CT_ENC_CNV_NORM_W, i); l.conv_norm_b = TF(CT_ENC_CNV_NORM_B, i);
        l.conv_pw1_w  = TF(CT_ENC_CNV_PW1_W,  i); l.conv_pw1_b  = TF(CT_ENC_CNV_PW1_B,  i);
        l.conv_dw_w   = TF(CT_ENC_CNV_DW_W,   i); l.conv_dw_b   = TF(CT_ENC_CNV_DW_B,   i);
        l.conv_bn_w   = TF(CT_ENC_CNV_BN_W,   i); l.conv_bn_b   = TF(CT_ENC_CNV_BN_B,   i);
        l.conv_bn_mean = TF(CT_ENC_CNV_BN_MEAN, i);
        l.conv_bn_var  = TF(CT_ENC_CNV_BN_VAR,  i);
        l.conv_pw2_w  = TF(CT_ENC_CNV_PW2_W,  i); l.conv_pw2_b  = TF(CT_ENC_CNV_PW2_B,  i);
        l.ff2_norm_w  = TF(CT_ENC_FF2_NORM_W, i); l.ff2_norm_b  = TF(CT_ENC_FF2_NORM_B, i);
        l.ff2_up_w    = TF(CT_ENC_FF2_UP_W,   i); l.ff2_up_b    = TF(CT_ENC_FF2_UP_B,   i);
        l.ff2_dn_w    = TF(CT_ENC_FF2_DN_W,   i); l.ff2_dn_b    = TF(CT_ENC_FF2_DN_B,   i);
        l.out_norm_w  = TF(CT_ENC_OUT_NORM_W, i); l.out_norm_b  = TF(CT_ENC_OUT_NORM_B, i);
    }

    m.enc_proj_w = T(CT_ENC_PROJ_W);  m.enc_proj_b = T(CT_ENC_PROJ_B);

    m.dec_emb_w    = T(CT_DEC_EMB_W);
    m.dec_pos_w    = T(CT_DEC_POS_W);
    m.dec_emb_ln_w = T(CT_DEC_EMB_LN_W);  m.dec_emb_ln_b = T(CT_DEC_EMB_LN_B);

    m.dec_layers.resize(hp.dec_n_layers);
    for (int i = 0; i < hp.dec_n_layers; i++) {
        auto & l = m.dec_layers[i];
        l.attn_ln_w = TF(CT_DEC_ATTN_LN_W, i); l.attn_ln_b = TF(CT_DEC_ATTN_LN_B, i);
        l.attn_q_w  = TF(CT_DEC_ATTN_Q_W,  i); l.attn_q_b  = TF(CT_DEC_ATTN_Q_B,  i);
        l.attn_k_w  = TF(CT_DEC_ATTN_K_W,  i); l.attn_k_b  = TF(CT_DEC_ATTN_K_B,  i);
        l.attn_v_w  = TF(CT_DEC_ATTN_V_W,  i); l.attn_v_b  = TF(CT_DEC_ATTN_V_B,  i);
        l.attn_o_w  = TF(CT_DEC_ATTN_O_W,  i); l.attn_o_b  = TF(CT_DEC_ATTN_O_B,  i);
        l.cross_ln_w = TF(CT_DEC_XATTN_LN_W, i); l.cross_ln_b = TF(CT_DEC_XATTN_LN_B, i);
        l.cross_q_w  = TF(CT_DEC_XATTN_Q_W,  i); l.cross_q_b  = TF(CT_DEC_XATTN_Q_B,  i);
        l.cross_k_w  = TF(CT_DEC_XATTN_K_W,  i); l.cross_k_b  = TF(CT_DEC_XATTN_K_B,  i);
        l.cross_v_w  = TF(CT_DEC_XATTN_V_W,  i); l.cross_v_b  = TF(CT_DEC_XATTN_V_B,  i);
        l.cross_o_w  = TF(CT_DEC_XATTN_O_W,  i); l.cross_o_b  = TF(CT_DEC_XATTN_O_B,  i);
        l.ffn_ln_w  = TF(CT_DEC_FFN_LN_W,  i); l.ffn_ln_b  = TF(CT_DEC_FFN_LN_B,  i);
        l.ffn_up_w  = TF(CT_DEC_FFN_UP_W,  i); l.ffn_up_b  = TF(CT_DEC_FFN_UP_B,  i);
        l.ffn_dn_w  = TF(CT_DEC_FFN_DN_W,  i); l.ffn_dn_b  = TF(CT_DEC_FFN_DN_B,  i);
    }

    m.dec_out_ln_w = T(CT_DEC_OUT_LN_W);  m.dec_out_ln_b = T(CT_DEC_OUT_LN_B);
    m.dec_head_w   = T(CT_DEC_HEAD_W);    m.dec_head_b   = T(CT_DEC_HEAD_B);

    return true;
}

// ---------------------------------------------------------------------------
// Utility: get F32 value from a ggml tensor (any dtype)
// ---------------------------------------------------------------------------

static float ct_f32(const ggml_tensor * t, int i0, int i1 = 0, int i2 = 0, int i3 = 0) {
    return ggml_get_f32_nd(const_cast<ggml_tensor*>(t), i0, i1, i2, i3);
}

// ---------------------------------------------------------------------------
// Layer norm (on float buffer, in-place)
// ---------------------------------------------------------------------------

static void ct_layer_norm(float * out, const float * in, int n,
                          const float * w, const float * b, float eps = 1e-5f) {
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = in[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < n; i++) out[i] = (in[i] - (float)mean) * inv * w[i] + b[i];
}

// ---------------------------------------------------------------------------
// SILU / Swish activation
// ---------------------------------------------------------------------------

static void ct_swish_inplace(float * x, int n) {
    for (int i = 0; i < n; i++) x[i] = swish(x[i]);
}

// ---------------------------------------------------------------------------
// Fast F16→F32 conversion using AVX2/F16C hardware instructions.
// Writes n floats into dst. Falls back to scalar bit-manipulation if no F16C.
// ---------------------------------------------------------------------------
static void ct_f16_to_f32(const uint16_t * src, float * dst, int n) {
#ifdef CT_HAVE_F16C
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
        __m256  f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t r;
        if      (exp == 0 && mant == 0) r = sign;
        else if (exp == 31)             r = sign | 0x7F800000u | (mant << 13);
        else                            r = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        memcpy(dst + i, &r, sizeof(float));
    }
#else
    for (int i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t r;
        if      (exp == 0 && mant == 0) r = sign;
        else if (exp == 31)             r = sign | 0x7F800000u | (mant << 13);
        else                            r = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        memcpy(dst + i, &r, sizeof(float));
    }
#endif
}

// Thread-local F32 buffer for on-the-fly F16→F32 weight conversion.
// Grows to max weight tensor size (~26 MB for enc ffn); never freed.
static thread_local std::vector<float> tl_w_buf;

// Convert an F16 ggml_tensor's data to F32 in tl_w_buf; return pointer.
// Valid until the next call to ct_tensor_f32() on the same thread.
static const float * ct_tensor_f32(const ggml_tensor * t) {
    int n = (int)ggml_nelements(t);
    if ((int)tl_w_buf.size() < n) tl_w_buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        ct_f16_to_f32((const uint16_t *)t->data, tl_w_buf.data(), n);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(tl_w_buf.data(), t->data, (size_t)n * sizeof(float));
    } else {
        for (int i = 0; i < n; i++) tl_w_buf[i] = ggml_get_f32_1d(const_cast<ggml_tensor *>(t), i);
    }
    return tl_w_buf.data();
}

// ---------------------------------------------------------------------------
// Linear layer: out[m, T] = w[m, n] × in[n, T]  (weight in row-major out×in)
// Returns newly allocated buffer (caller frees).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Feature extraction: raw PCM → log-mel spectrogram
// Returns float array of shape (n_mels, T_mel), row-major.
// ---------------------------------------------------------------------------

static std::vector<float> cohere_compute_features(const cohere_hparams & hp,
                                                   const float * fe_mel_fb_data,
                                                   const float * fe_window_data,
                                                   const float * samples, int n_samples,
                                                   int & T_out) {
    const int n_fft     = hp.n_fft;
    const int hop       = hp.hop_length;
    const int win       = hp.win_length;
    const int n_freqs   = hp.n_freqs();
    const int n_mels    = hp.n_mels;
    const float preemph = 0.97f;
    const float log_grd = (float)(1.0 / (1 << 24));

    // Pre-emphasis
    std::vector<float> pe(n_samples);
    pe[0] = samples[0];
    for (int i = 1; i < n_samples; i++) pe[i] = samples[i] - preemph * samples[i-1];

    // Center-pad
    int pad = n_fft / 2;
    std::vector<float> padded(pad + n_samples + pad, 0.0f);
    memcpy(padded.data() + pad, pe.data(), n_samples * sizeof(float));

    // Number of frames
    int n_pad = (int)padded.size();
    int T = (n_pad - n_fft) / hop + 1;
    T_out = T;

    // Hann window (from fe_window tensor, length win_length, padded to n_fft)
    std::vector<float> window(n_fft, 0.0f);
    int lpad = (n_fft - win) / 2;
    for (int i = 0; i < win; i++) window[lpad + i] = fe_window_data[i];

    // STFT → power spectrum → mel → log → normalize
    // Use FFTW3f real-to-complex FFT: O(n_fft·log n_fft) vs O(n_fft²) for direct DFT.
    std::vector<float> power(n_freqs * T, 0.0f);
    {
        std::vector<float>           fft_in(n_fft);
        std::vector<fftwf_complex>   fft_out(n_freqs);
        fftwf_plan plan = fftwf_plan_dft_r2c_1d(n_fft, fft_in.data(), fft_out.data(), FFTW_ESTIMATE);
        for (int t = 0; t < T; t++) {
            const float * frame = padded.data() + t * hop;
            for (int n = 0; n < n_fft; n++) fft_in[n] = frame[n] * window[n];
            fftwf_execute(plan);
            for (int k = 0; k < n_freqs; k++) {
                float re = fft_out[k][0], im = fft_out[k][1];
                power[t * n_freqs + k] = re*re + im*im;
            }
        }
        fftwf_destroy_plan(plan);
    }

    // mel filterbank: fe_mel_fb shape [1, n_mels, n_freqs]
    // power [T, n_freqs] @ fe_mel_fb^T [n_freqs, n_mels] = mel [T, n_mels]
    std::vector<float> mel(n_mels * T, 0.0f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, n_mels, n_freqs,
                1.0f, power.data(), n_freqs, fe_mel_fb_data, n_freqs,
                0.0f, mel.data(), n_mels);

    for (int i = 0; i < T * n_mels; i++) {
        mel[i] = logf(mel[i] + log_grd);
    }

    // Per-feature normalization: biased std (matches ONNX: std = sqrt(mean(diff²)))
    for (int m = 0; m < n_mels; m++) {
        double mean = 0.0, var = 0.0;
        for (int t = 0; t < T; t++) mean += mel[t * n_mels + m];
        mean /= T;
        for (int t = 0; t < T; t++) { double d = mel[t * n_mels + m] - mean; var += d*d; }
        float std = sqrtf((float)(var / T + 1e-5));
        for (int t = 0; t < T; t++) mel[t * n_mels + m] = (mel[t * n_mels + m] - (float)mean) / std;
    }

    cohere_debug("cohere: mel m=0, t=0..4: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           mel[0*n_mels + 0], mel[1*n_mels + 0], mel[2*n_mels + 0], mel[3*n_mels + 0], mel[4*n_mels + 0]);
    cohere_debug("cohere: mel t=0, m=0..4: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           mel[0*n_mels + 0], mel[0*n_mels + 1], mel[0*n_mels + 2], mel[0*n_mels + 3], mel[0*n_mels + 4]);

    return mel; // shape: [n_mels, T], time-major
}

// ---------------------------------------------------------------------------
// Conformer: sinusoidal relative positional encoding
// Returns (2T-1, d_model) array, positions from T-1 to -(T-1)
// ---------------------------------------------------------------------------

static std::vector<float> ct_rel_pos_enc(int T, int d_model) {
    int n_pos = 2 * T - 1;
    std::vector<float> pe(n_pos * d_model, 0.0f);
    for (int i = 0; i < n_pos; i++) {
        float pos = (float)(T - 1 - i); // T-1 down to -(T-1)
        for (int j = 0; j < d_model / 2; j++) {
            float div = powf(10000.0f, 2.0f * j / d_model);
            pe[i * d_model + 2*j]   = sinf(pos / div);
            pe[i * d_model + 2*j+1] = cosf(pos / div);
        }
    }
    return pe;
}

// ---------------------------------------------------------------------------
// Pre-compute cross-attention K and V for all decoder layers.
// Call once per utterance after encoding; results stored in cross_kv_k/v.
// Layout: cross_kv_k[li] has shape (T_enc, dec_d_model), row-major.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Decoder Graph Builder
// ---------------------------------------------------------------------------

static struct ggml_cgraph * cohere_build_graph_decoder(
    struct cohere_context * ctx,
    const int * tokens, int n_tokens, int offset
) {
    const auto & model = ctx->model;
    const auto & hp    = model.hparams;
    const int d        = hp.dec_d_model;
    const int n_heads  = hp.dec_n_heads;
    const int head_dim = hp.dec_head_dim;
    const int max_ctx  = hp.dec_max_ctx;

    // For single-token autoregressive steps, use fixed-size KV views (max_ctx) so that
    // the ggml_gallocr can reuse its allocation plan across steps instead of re-reserving
    // each time the shape changes. An additive mask zeros out unwritten cache slots.
    const int kv_len = (n_tokens == 1) ? max_ctx : (offset + n_tokens);

    cohere_debug("\n--- cohere_build_graph_decoder: n_tokens=%d, offset=%d kv_len=%d ---\n", n_tokens, offset, kv_len);

    struct ggml_init_params params = {
        .mem_size   = ctx->compute_meta.size(),
        .mem_buffer = ctx->compute_meta.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(embd, "embd");
    ggml_set_input(embd);

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    // [d, n_tokens]
    struct ggml_tensor * cur = ggml_add(ctx0,
        ggml_get_rows(ctx0, model.dec_emb_w, embd),
        ggml_get_rows(ctx0, model.dec_pos_w, position)
    );
    cohere_log_tensor("input_embd", cur);

    // emb_ln
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, model.dec_emb_ln_w);
        cur = ggml_add_inplace(ctx0, cur, model.dec_emb_ln_b);
    cohere_log_tensor("emb_ln", cur);

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto & layer = model.dec_layers[il];
        struct ggml_tensor * inpL = cur;

        cur = ggml_norm(ctx0, inpL, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.attn_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.attn_ln_b);

        // self-attention Q, K, V
        struct ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_q_w, cur), layer.attn_q_b);
        struct ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_k_w, cur), layer.attn_k_b);
        struct ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_v_w, cur), layer.attn_v_b);

        if (il == 0) {
            cohere_log_tensor("Qcur", Qcur);
            cohere_log_tensor("Kcur", Kcur);
            cohere_log_tensor("Vcur", Vcur);
        }

        // Store current K, V into cache
        {
            struct ggml_tensor * k_view = ggml_view_4d(ctx0, ctx->kv_k,
                head_dim, n_tokens, n_heads, 1,
                ctx->kv_k->nb[1], ctx->kv_k->nb[2], ctx->kv_k->nb[3],
                il * ctx->kv_k->nb[3] + offset * ctx->kv_k->nb[1]);
            
            struct ggml_tensor * v_view = ggml_view_4d(ctx0, ctx->kv_v,
                head_dim, n_tokens, n_heads, 1,
                ctx->kv_v->nb[1], ctx->kv_v->nb[2], ctx->kv_v->nb[3],
                il * ctx->kv_v->nb[3] + offset * ctx->kv_v->nb[1]);

            struct ggml_tensor * K_perm = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            struct ggml_tensor * V_perm = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Vcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);


            ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_view));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_view));
        }

        struct ggml_tensor * Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3); // [hd, n_tok, n_heads]
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, ctx->kv_k,
            head_dim, kv_len, n_heads,
            ctx->kv_k->nb[1], ctx->kv_k->nb[2],
            il * ctx->kv_k->nb[3]); // [hd, kv_len, n_heads]
        K = ggml_cont(ctx0, K);
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q); // [kv_len, n_tok, n_heads]
        KQ = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrtf((float)head_dim));
        if (n_tokens == 1) {
            // Additive mask: 0.0 for valid positions, -inf for unwritten slots
            KQ = ggml_add_inplace(ctx0, KQ, ctx->kq_mask);
        } else {
            KQ = ggml_diag_mask_inf_inplace(ctx0, KQ, offset);
        }
        KQ = ggml_soft_max_inplace(ctx0, KQ);

        struct ggml_tensor * V = ggml_view_3d(ctx0, ctx->kv_v,
            head_dim, kv_len, n_heads,
            ctx->kv_v->nb[1], ctx->kv_v->nb[2],
            il * ctx->kv_v->nb[3]); // [hd, kv_len, n_heads]

        struct ggml_tensor * V_trans = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3)); // [offset+n_tok, hd, n_heads]
        struct ggml_tensor * sa_out = ggml_mul_mat(ctx0, V_trans, KQ); // [hd, n_tok, n_heads]

        sa_out = ggml_permute(ctx0, sa_out, 0, 2, 1, 3); // [hd, n_heads, n_tok]
        sa_out = ggml_cont(ctx0, sa_out);
        cur = ggml_reshape_2d(ctx0, sa_out, d, n_tokens);

        // out projection
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_o_w, cur), layer.attn_o_b);
        cur = ggml_add(ctx0, cur, inpL);
        if (il == 0) ggml_set_name(cur, "h0_after_sa");

        struct ggml_tensor * inpCA = cur;

        // cross-attention
        cur = ggml_norm(ctx0, inpCA, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.cross_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.cross_ln_b);

        struct ggml_tensor * CQ = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_q_w, cur), layer.cross_q_b);
        CQ = ggml_reshape_3d(ctx0, CQ, head_dim, n_heads, n_tokens);
        CQ = ggml_permute(ctx0, CQ, 0, 2, 1, 3); // [hd, n_tok, n_heads]

        struct ggml_tensor * CK = ctx->cross_kv_k[il]; // [hd, T_enc, n_heads]
        struct ggml_tensor * CV_trans = ctx->cross_kv_v[il]; // [T_enc, hd, n_heads]

        struct ggml_tensor * C_KQ = ggml_mul_mat(ctx0, CK, CQ); // [T_enc, n_tokens, n_heads]
        C_KQ = ggml_scale_inplace(ctx0, C_KQ, 1.0f / sqrtf((float)head_dim));
        C_KQ = ggml_soft_max_inplace(ctx0, C_KQ);

        struct ggml_tensor * ca_out = ggml_mul_mat(ctx0, CV_trans, C_KQ); // [hd, n_tok, n_heads]

        ca_out = ggml_permute(ctx0, ca_out, 0, 2, 1, 3); // [hd, n_heads, n_tok]
        ca_out = ggml_cont(ctx0, ca_out);
        cur = ggml_reshape_2d(ctx0, ca_out, d, n_tokens);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_o_w, cur), layer.cross_o_b);
        cur = ggml_add(ctx0, cur, inpCA);
        if (il == 0) ggml_set_name(cur, "h0_after_ca");

        struct ggml_tensor * inpFFN = cur;

        // FFN
        cur = ggml_norm(ctx0, inpFFN, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.ffn_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.ffn_ln_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_up_w, cur), layer.ffn_up_b);
        cur = ggml_relu_inplace(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_dn_w, cur), layer.ffn_dn_b);


        cur = ggml_add(ctx0, cur, inpFFN);

        if (il == 0) {
            cohere_log_tensor("layer_0_out", cur);
        }
    }

    // final output norm
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul_inplace(ctx0, cur, model.dec_out_ln_w);
    cur = ggml_add_inplace(ctx0, cur, model.dec_out_ln_b);


    // logits
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.dec_head_w, cur), model.dec_head_b);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ---------------------------------------------------------------------------
// Decoder: one step (auto-regressive)
// tokens:   [offset .. offset+n_tok-1]
// Returns logits: (n_tok, vocab_size)
// ---------------------------------------------------------------------------

static std::vector<float> cohere_decode_step(
    struct cohere_context * ctx,
    int T_enc,
    const int * tokens, int n_tok, int offset
) {
    const auto & hp = ctx->model.hparams;
    const int vocab_size = hp.vocab_size;
    auto & perf = ctx->perf;
    int64_t t0, t1;

    ctx->cached_T_enc = T_enc;

    t0 = ggml_time_us();
    struct ggml_cgraph * gf = cohere_build_graph_decoder(ctx, tokens, n_tok, offset);
    t1 = ggml_time_us();
    perf.t_dec_build_us += (t1 - t0);
    if      (perf.n_dec_steps == 0) perf.dec_n_nodes_prompt = ggml_graph_n_nodes(gf);
    else if (perf.n_dec_steps == 1) perf.dec_n_nodes_step   = ggml_graph_n_nodes(gf);

    t0 = ggml_time_us();
    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        fprintf(stderr, "cohere: failed to allocate decoder graph\n");
        return {};
    }
    t1 = ggml_time_us();
    perf.t_dec_alloc_us += (t1 - t0);

    // Set inputs
    struct ggml_tensor * embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, tokens, 0, n_tok * sizeof(int));

    struct ggml_tensor * position = ggml_graph_get_tensor(gf, "position");
    std::vector<int> pos_data(n_tok);
    for (int i = 0; i < n_tok; i++) pos_data[i] = offset + i;
    ggml_backend_tensor_set(position, pos_data.data(), 0, n_tok * sizeof(int));

    // For single-token steps, update the KQ mask: 0.0 for valid positions, -inf for the rest.
    // This enables fixed-size KV views (kv_len=max_ctx) so the gallocr plan can be reused.
    if (n_tok == 1) {
        const int max_ctx = ctx->model.hparams.dec_max_ctx;
        const int valid_len = offset + n_tok;
        std::vector<float> mask_data(max_ctx, -INFINITY);
        for (int i = 0; i < valid_len && i < max_ctx; i++) mask_data[i] = 0.0f;
        ggml_backend_tensor_set(ctx->kq_mask, mask_data.data(), 0, max_ctx * sizeof(float));
    }

    t0 = ggml_time_us();
    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf, ctx->params.n_threads)) {
        fprintf(stderr, "cohere: failed to compute decoder graph\n");
        return {};
    }
    t1 = ggml_time_us();
    {
        int64_t dt = t1 - t0;
        perf.t_dec_compute_us += dt;
        if (dt < perf.t_dec_step_min_us) perf.t_dec_step_min_us = dt;
        if (dt > perf.t_dec_step_max_us) perf.t_dec_step_max_us = dt;
    }

    // Extract logits (last node in graph)
    struct ggml_tensor * logits_t = ggml_graph_node(gf, -1);
    std::vector<float> logits(n_tok * vocab_size);
    t0 = ggml_time_us();
    ggml_backend_tensor_get(logits_t, logits.data(), 0, logits.size() * sizeof(float));
    t1 = ggml_time_us();
    perf.t_dec_logits_us += (t1 - t0);

    if (offset == 0) {
        // Log some interesting hidden states for the first step (prompt)
        struct ggml_tensor * emb_ln = ggml_graph_get_tensor(gf, "emb_ln");
        if (emb_ln) {
            std::vector<float> h(ggml_nelements(emb_ln));
            ggml_backend_tensor_get(emb_ln, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after emb_ln: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                h[0], h[1], h[2], h[3], h[4]);
        }

        struct ggml_tensor * h0_sa = ggml_graph_get_tensor(gf, "h0_after_sa");
        if (h0_sa) {
            std::vector<float> h(ggml_nelements(h0_sa));
            ggml_backend_tensor_get(h0_sa, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after sa_li0: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                h[0], h[1], h[2], h[3], h[4]);
        }

        struct ggml_tensor * h0_ca = ggml_graph_get_tensor(gf, "h0_after_ca");
        if (h0_ca) {
            std::vector<float> h(ggml_nelements(h0_ca));
            ggml_backend_tensor_get(h0_ca, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after ca_li0: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                h[0], h[1], h[2], h[3], h[4]);
        }
        
        // Log top logits for last token of prompt
        const float * last_logits = logits.data() + (n_tok - 1) * vocab_size;
        int top_id = 0;
        for (int i = 1; i < vocab_size; i++) if (last_logits[i] > last_logits[top_id]) top_id = i;
        cohere_debug("cohere: step 0 (prompt) top_tok: %d, logit: %.4f, tok_749_logit: %.4f\n", 
               top_id, last_logits[top_id], last_logits[749]);
    }

    return logits;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

// Pre-populate the ct_to_f32_ref cache for every model weight tensor.
// Called once at init so all inferences pay zero conversion cost.
static void cohere_model_warm_cache(const cohere_model & m) {
    (void)m;
}

struct cohere_context_params cohere_context_default_params(void) {
    return { .n_threads = 4, .use_flash = false };
}

struct cohere_context * cohere_init_from_file(const char * path_model,
                                              struct cohere_context_params params) {
    ggml_time_init();
    auto * ctx = new cohere_context;
    ctx->params = params;

    // Thread count: overridable via COHERE_THREADS env var.
    // NOTE: ggml_backend_cpu_set_n_threads is NOT called by default —
    // profiling showed it regressed perf for our small matrix sizes.
    {
        const char * env = getenv("COHERE_THREADS");
        if (env) {
            int n = atoi(env);
            if (n > 0) params.n_threads = n;
        }
    }

    // ---------------------------------------------------------------------------
    // Backend selection: GPU first (Metal/CUDA), CPU fallback.
    // Set COHERE_DEVICE=cpu  to force CPU.
    // Set COHERE_DEVICE=metal (or cuda) to request a specific backend.
    // ---------------------------------------------------------------------------
    ggml_backend_load_all(); // registers Metal (macOS), CUDA, Vulkan, etc. if compiled in

    {
        const char * dev_env = getenv("COHERE_DEVICE");
        if (dev_env && strlen(dev_env) > 0) {
            ctx->ggml_backend = ggml_backend_init_by_name(dev_env, nullptr);
            if (!ctx->ggml_backend) {
                fprintf(stderr, "cohere: WARNING: COHERE_DEVICE='%s' not available, falling back\n", dev_env);
            }
        }
        if (!ctx->ggml_backend) {
            ctx->ggml_backend = ggml_backend_init_best(); // GPU > CPU
        }
        if (!ctx->ggml_backend) {
            fprintf(stderr, "cohere: failed to initialize any ggml backend\n");
            delete ctx;
            return nullptr;
        }
    }

    // Always have a CPU backend available as scheduler fallback for unsupported ops
    ctx->ggml_backend_cpu = ggml_backend_cpu_init();
    bool using_gpu = !ggml_backend_is_cpu(ctx->ggml_backend);
    fprintf(stderr, "cohere: backend: %s%s\n",
            ggml_backend_name(ctx->ggml_backend),
            using_gpu ? "" : " (CPU-only)");

    // Apply thread count only when explicitly requested via env var
    if (getenv("COHERE_THREADS")) {
        fprintf(stderr, "cohere: applying n_threads=%d to CPU backend [COHERE_THREADS override]\n",
                params.n_threads);
        ggml_backend_cpu_set_n_threads(ctx->ggml_backend_cpu, params.n_threads);
        if (!using_gpu) {
            ggml_backend_cpu_set_n_threads(ctx->ggml_backend, params.n_threads);
        }
    } else {
        fprintf(stderr, "cohere: n_threads param=%d (use COHERE_THREADS=N to override)\n",
                params.n_threads);
    }

    if (!cohere_load_model(ctx->model, ctx->vocab, path_model, ctx->ggml_backend)) {
        cohere_free(ctx);
        return nullptr;
    }

    // Fold inference BN into depthwise conv weights — removes 6 graph nodes × 48 layers
    cohere_fold_batchnorm(ctx->model);

    const auto & hp = ctx->model.hparams;

    // Allocate persistent KV cache
    {
        struct ggml_init_params kv_params = {
            .mem_size   = ggml_tensor_overhead() * 3 + 1024,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        ctx->kv_ctx  = ggml_init(kv_params);
        ctx->kv_k    = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F32, hp.dec_head_dim, hp.dec_max_ctx, hp.dec_n_heads, hp.dec_n_layers);
        ctx->kv_v    = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F32, hp.dec_head_dim, hp.dec_max_ctx, hp.dec_n_heads, hp.dec_n_layers);
        ctx->kq_mask = ggml_new_tensor_1d(ctx->kv_ctx, GGML_TYPE_F32, hp.dec_max_ctx);
        ggml_set_name(ctx->kq_mask, "kq_mask");
        ggml_backend_buffer_t kv_buf = ggml_backend_alloc_buffer(ctx->ggml_backend,
            ggml_nbytes(ctx->kv_k) + ggml_nbytes(ctx->kv_v) + ggml_nbytes(ctx->kq_mask));
        char * base = (char *)ggml_backend_buffer_get_base(kv_buf);
        ggml_backend_tensor_alloc(kv_buf, ctx->kv_k,    (void *)(base));
        ggml_backend_tensor_alloc(kv_buf, ctx->kv_v,    (void *)(base + ggml_nbytes(ctx->kv_k)));
        ggml_backend_tensor_alloc(kv_buf, ctx->kq_mask, (void *)(base + ggml_nbytes(ctx->kv_k) + ggml_nbytes(ctx->kv_v)));

        fprintf(stderr, "cohere: kv cache     = %.1f MiB  (dec_head_dim=%d max_ctx=%d n_heads=%d n_layers=%d)\n",
                ggml_backend_buffer_get_size(kv_buf) / 1048576.0,
                hp.dec_head_dim, hp.dec_max_ctx, hp.dec_n_heads, hp.dec_n_layers);
    }

    // Initialize scheduler — GPU backend first (highest priority), CPU as fallback
    if (using_gpu) {
        ggml_backend_t backends[] = { ctx->ggml_backend, ctx->ggml_backend_cpu };
        ctx->ggml_alloc = ggml_backend_sched_new(backends, nullptr, 2, 16384, false, false);
    } else {
        ggml_backend_t backends[] = { ctx->ggml_backend };
        ctx->ggml_alloc = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);
    }

    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + 1024);

    // Create constants context
    ctx->ctx_const = ggml_init({ .mem_size = 1024, .mem_buffer = nullptr, .no_alloc = false });
    ctx->eps = ggml_new_tensor_1d(ctx->ctx_const, GGML_TYPE_F32, 1);
    float eps_val = 1e-5f;
    memcpy(ctx->eps->data, &eps_val, sizeof(float));

    ctx->cross_kv_ctx = nullptr;
    ctx->cross_kv_buf = nullptr;

    // Log static memory layout
    if (ctx->model.buf) {
        fprintf(stderr, "cohere: model weights = %.1f MiB\n",
                ggml_backend_buffer_get_size(ctx->model.buf) / 1048576.0);
    }
    fprintf(stderr, "cohere: compute_meta  = %.1f KiB\n", ctx->compute_meta.size() / 1024.0);

    return ctx;
}

void cohere_free(struct cohere_context * ctx) {
    if (!ctx) return;
    if (ctx->ggml_alloc)        ggml_backend_sched_free(ctx->ggml_alloc);
    if (ctx->ggml_backend)      ggml_backend_free(ctx->ggml_backend);
    if (ctx->ggml_backend_cpu && ctx->ggml_backend_cpu != ctx->ggml_backend)
        ggml_backend_free(ctx->ggml_backend_cpu);
    if (ctx->kv_ctx)        ggml_free(ctx->kv_ctx);
    if (ctx->ctx_const)     ggml_free(ctx->ctx_const);
    if (ctx->cross_kv_ctx)  ggml_free(ctx->cross_kv_ctx);
    if (ctx->cross_kv_buf)  ggml_backend_buffer_free(ctx->cross_kv_buf);
    if (ctx->model.buf)     ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)     ggml_free(ctx->model.ctx);
    delete ctx;
}

int cohere_n_vocab(struct cohere_context * ctx) {
    return ctx->vocab.n_vocab();
}

const char * cohere_token_to_str(struct cohere_context * ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size()) return "<unk>";
    return ctx->vocab.id_to_token[id].c_str();
}

int cohere_str_to_token(struct cohere_context * ctx, const char * s) {
    return ctx->vocab.token_id(s);
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

static std::vector<float> ct_get_f32(const ggml_tensor * t) {
    const int n = (int)ggml_nelements(t);
    std::vector<float> res(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, res.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int i = 0; i < n; i++) res[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        fprintf(stderr, "ct_get_f32: unsupported type %d\n", (int)t->type);
        abort();
    }
    return res;
}

// ---------------------------------------------------------------------------
// BatchNorm folding — called once after model load.
//
// Inference-time BN is: y = (x - mean) / sqrt(var + eps) * scale + bn_b
// Equivalently:         y = x * s + (bn_b - mean * s)   where s = scale/sqrt(var+eps)
//
// Since mean/var are fixed constants after training, we fold s into the
// depthwise conv kernel weights and absorb the full bias shift into conv_dw_b:
//
//   conv_dw_w[:, c] *= s[c]
//   conv_dw_b[c]     = (conv_dw_b[c] - mean[c]) * s[c] + bn_b[c]
//
// After this the encoder graph drops the 6-node BN block (288 nodes total for
// 48 layers) and replaces it with nothing — the folded conv already applies it.
// ---------------------------------------------------------------------------
static void cohere_fold_batchnorm(cohere_model & model) {
    const int d   = model.hparams.enc_d_model;   // 1280
    const int k   = model.hparams.enc_conv_k;    // 9
    const float eps = 1e-5f;

    for (int il = 0; il < model.hparams.enc_n_layers; il++) {
        auto & layer = model.enc_layers[il];

        // Read all F32 BN parameters (all are F32 in the GGUF)
        std::vector<float> bn_mean(d), bn_var(d), bn_scale(d), bn_bias(d);
        ggml_backend_tensor_get(layer.conv_bn_mean, bn_mean.data(),  0, d * sizeof(float));
        ggml_backend_tensor_get(layer.conv_bn_var,  bn_var.data(),   0, d * sizeof(float));
        ggml_backend_tensor_get(layer.conv_bn_w,    bn_scale.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(layer.conv_bn_b,    bn_bias.data(),  0, d * sizeof(float));

        // Per-channel fold factor: s[c] = bn_scale[c] / sqrt(bn_var[c] + eps)
        std::vector<float> s(d);
        for (int c = 0; c < d; c++)
            s[c] = bn_scale[c] / sqrtf(bn_var[c] + eps);

        // Fold s into conv_dw_w (F16 tensor, ggml shape ne[0]=k, ne[1]=1, ne[2]=d)
        // Linear index for kernel pos ki, channel c: ki + c*k
        {
            std::vector<float> w_f32 = ct_get_f32(layer.conv_dw_w); // k*d elements
            for (int c = 0; c < d; c++)
                for (int ki = 0; ki < k; ki++)
                    w_f32[ki + c * k] *= s[c];
            // Write back as F16
            std::vector<ggml_fp16_t> w_f16(k * d);
            ggml_fp32_to_fp16_row(w_f32.data(), w_f16.data(), k * d);
            ggml_backend_tensor_set(layer.conv_dw_w, w_f16.data(), 0, k * d * sizeof(ggml_fp16_t));
        }

        // Fold into conv_dw_b (F32 tensor, shape [d]):
        // b_folded[c] = (dw_b[c] - mean[c]) * s[c] + bn_bias[c]
        {
            std::vector<float> dw_b(d);
            ggml_backend_tensor_get(layer.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
            for (int c = 0; c < d; c++)
                dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_bias[c];
            ggml_backend_tensor_set(layer.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
        }
    }
    fprintf(stderr, "cohere: BN folded into conv_dw weights for %d layers\n",
            model.hparams.enc_n_layers);
}

char * cohere_transcribe(struct cohere_context * ctx,
                         const float * samples, int n_samples,
                         const char * lang) {
    const auto & hp  = ctx->model.hparams;
    const auto & voc = ctx->vocab;
    const int n_heads  = hp.dec_n_heads;
    const int head_dim = hp.dec_head_dim;
    auto & perf = ctx->perf;
    int64_t t0, t1;

    cohere_perf_reset(perf);
    int64_t t_total_start = ggml_time_us();

    fprintf(stderr, "cohere: transcribe started   n_samples=%d  audio=%.2fs\n",
            n_samples, (double)n_samples / hp.sample_rate);

    // --- Feature extraction ---
    auto mel_fb = ct_get_f32(ctx->model.fe_mel_fb);
    auto window = ct_get_f32(ctx->model.fe_window);

    // --- Feature extraction + Encoder ---
    // For audio > 30s: process in 30s chunks to cap O(T²) attention cost.
    // Cross-KV (K/V projections of encoder output) is extracted from each chunk's
    // encoder graph and scatter-copied into the final K[head_dim,T_total,n_heads] /
    // V[T_total,head_dim,n_heads] layout on the CPU, then uploaded to the backend.
    const int CHUNK_SAMPLES = 30 * hp.sample_rate;
    const bool do_chunked = (n_samples > CHUNK_SAMPLES);

    int T_enc_total = 0;

    // Per-chunk K/V CPU storage: partial_k[il][chunk] and partial_v[il][chunk]
    // K chunk: [head_dim, T_c, n_heads] layout (raw F32 bytes from encoder graph)
    // V chunk: [T_c, head_dim, n_heads] layout (raw F32 bytes from encoder graph)
    std::vector<std::vector<std::vector<float>>> partial_k(hp.dec_n_layers);
    std::vector<std::vector<std::vector<float>>> partial_v(hp.dec_n_layers);
    std::vector<int> T_enc_chunks;
    for (int il = 0; il < hp.dec_n_layers; il++) {
        partial_k[il].reserve(4);
        partial_v[il].reserve(4);
    }

    // Optional per-op profiling (COHERE_PROF=1, single-chunk only)
    cohere_prof_state prof_state;
    bool do_prof = !do_chunked && (getenv("COHERE_PROF") != nullptr);

    {
        int n_chunks = 0;
        for (int sample_offset = 0; sample_offset < n_samples; sample_offset += CHUNK_SAMPLES) {
            int chunk_n = std::min(CHUNK_SAMPLES, n_samples - sample_offset);
            n_chunks++;

            // Feature extraction for this chunk
            int T_mel_c = 0;
            t0 = ggml_time_us();
            auto mel_c = cohere_compute_features(hp, mel_fb.data(), window.data(),
                                                  samples + sample_offset, chunk_n, T_mel_c);
            t1 = ggml_time_us();
            perf.t_features_us += (t1 - t0);

            if (do_chunked) {
                fprintf(stderr, "cohere: chunk %d/%d  n=%d  T_mel=%d\n",
                        n_chunks, (n_samples + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES, chunk_n, T_mel_c);
            } else {
                fprintf(stderr, "cohere: features done         T_mel=%d  %.1f ms\n",
                        T_mel_c, perf.t_features_us / 1e3);
            }

            // Build encoder graph for this chunk
            t0 = ggml_time_us();
            struct ggml_cgraph * gf_enc = cohere_build_graph_encoder(ctx, T_mel_c);
            t1 = ggml_time_us();
            perf.t_enc_build_us += (t1 - t0);
            perf.enc_n_nodes = ggml_graph_n_nodes(gf_enc);

            if (!do_chunked) {
                fprintf(stderr, "cohere: enc graph built       nodes=%d  %.1f ms\n",
                        perf.enc_n_nodes, perf.t_enc_build_us / 1e3);
            }

            // Allocate
            t0 = ggml_time_us();
            ggml_backend_sched_reset(ctx->ggml_alloc);
            if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf_enc)) {
                fprintf(stderr, "cohere: failed to allocate encoder graph (chunk %d)\n", n_chunks);
                return nullptr;
            }
            t1 = ggml_time_us();
            perf.t_enc_alloc_us += (t1 - t0);
            perf.mem_sched_buf = ggml_backend_sched_get_buffer_size(ctx->ggml_alloc, ctx->ggml_backend);
            if (!do_chunked) {
                fprintf(stderr, "cohere: enc sched alloc       %.1f ms   sched_buf=%.1f MiB\n",
                        perf.t_enc_alloc_us / 1e3, perf.mem_sched_buf / 1048576.0);
            }

            // Set inputs
            struct ggml_tensor * mel_t = ggml_graph_get_tensor(gf_enc, "mel");
            if (!mel_t) { fprintf(stderr, "error: mel tensor not found\n"); return nullptr; }
            ggml_backend_tensor_set(mel_t, mel_c.data(), 0, mel_c.size() * sizeof(float));

            int H1c = (T_mel_c + 2 - 3) / 2 + 1;
            int H2c = (H1c + 2 - 3) / 2 + 1;
            int H3c = (H2c + 2 - 3) / 2 + 1;
            auto pos_enc_c = ct_rel_pos_enc(H3c, hp.enc_d_model);
            struct ggml_tensor * pos_enc_t = ggml_graph_get_tensor(gf_enc, "pos_enc");
            if (!pos_enc_t) { fprintf(stderr, "error: pos_enc tensor not found\n"); return nullptr; }
            ggml_backend_tensor_set(pos_enc_t, pos_enc_c.data(), 0, pos_enc_c.size() * sizeof(float));

            if (do_prof) {
                ggml_backend_sched_set_eval_callback(ctx->ggml_alloc, cohere_prof_eval_cb, &prof_state);
            }

            // Compute
            t0 = ggml_time_us();
            if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf_enc, ctx->params.n_threads)) {
                fprintf(stderr, "cohere: failed to compute encoder graph (chunk %d)\n", n_chunks);
                return nullptr;
            }
            t1 = ggml_time_us();
            perf.t_enc_compute_us += (t1 - t0);

            if (do_chunked) {
                fprintf(stderr, "cohere: chunk %d enc done     %.1f ms\n", n_chunks, (t1-t0)/1e3);
            } else {
                fprintf(stderr, "cohere: enc compute done      %.1f ms\n", perf.t_enc_compute_us / 1e3);
            }

            if (do_prof) {
                ggml_backend_sched_set_eval_callback(ctx->ggml_alloc, nullptr, nullptr);
            }

            // Extract T_enc for this chunk
            struct ggml_tensor * enc_out_t = ggml_graph_get_tensor(gf_enc, "enc_out");
            if (!enc_out_t) { fprintf(stderr, "error: enc_out tensor not found\n"); return nullptr; }
            int T_enc_c = enc_out_t->ne[1];
            T_enc_total += T_enc_c;
            T_enc_chunks.push_back(T_enc_c);

            // Extract cross-KV from this chunk's encoder graph into CPU vectors.
            // K shape: [head_dim, T_enc_c, n_heads] (raw F32 from encoder graph)
            // V shape: [T_enc_c, head_dim, n_heads] (raw F32 from encoder graph)
            for (int il = 0; il < hp.dec_n_layers; il++) {
                char ck_name[32], cv_name[32];
                snprintf(ck_name, sizeof(ck_name), "ck_%d", il);
                snprintf(cv_name, sizeof(cv_name), "cv_%d", il);
                struct ggml_tensor * ck_src = ggml_graph_get_tensor(gf_enc, ck_name);
                struct ggml_tensor * cv_src = ggml_graph_get_tensor(gf_enc, cv_name);
                if (!ck_src || !cv_src) {
                    fprintf(stderr, "error: cross-KV tensor %s or %s not found\n", ck_name, cv_name);
                    return nullptr;
                }
                partial_k[il].emplace_back(ggml_nelements(ck_src));
                partial_v[il].emplace_back(ggml_nelements(cv_src));
                ggml_backend_tensor_get(ck_src, partial_k[il].back().data(), 0, ggml_nbytes(ck_src));
                ggml_backend_tensor_get(cv_src, partial_v[il].back().data(), 0, ggml_nbytes(cv_src));
            }
        } // end chunk loop

        if (do_chunked) {
            fprintf(stderr, "cohere: enc compute done      %.1f ms  (chunked %d×30s)\n",
                    perf.t_enc_compute_us / 1e3, n_chunks);
        }

        if (do_prof) {
            cohere_prof_print(prof_state);
        }
    }

    // Assemble cross-KV from per-chunk CPU data and upload to backend buffer.
    {
        t0 = ggml_time_us();

        if (ctx->cross_kv_ctx) ggml_free(ctx->cross_kv_ctx);
        if (ctx->cross_kv_buf) ggml_backend_buffer_free(ctx->cross_kv_buf);

        ctx->cross_kv_k.resize(hp.dec_n_layers);
        ctx->cross_kv_v.resize(hp.dec_n_layers);

        struct ggml_init_params ckv_params = {
            .mem_size   = (hp.dec_n_layers * 2 + 10) * ggml_tensor_overhead(),
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        ctx->cross_kv_ctx = ggml_init(ckv_params);

        for (int il = 0; il < hp.dec_n_layers; il++) {
            ctx->cross_kv_k[il] = ggml_new_tensor_3d(ctx->cross_kv_ctx, GGML_TYPE_F32,
                                                       head_dim, T_enc_total, n_heads);
            ctx->cross_kv_v[il] = ggml_new_tensor_3d(ctx->cross_kv_ctx, GGML_TYPE_F32,
                                                       T_enc_total, head_dim, n_heads);
        }

        const size_t k_size = ggml_nbytes(ctx->cross_kv_k[0]); // head_dim × T_enc_total × n_heads × 4
        const size_t v_size = ggml_nbytes(ctx->cross_kv_v[0]); // T_enc_total × head_dim × n_heads × 4
        ctx->cross_kv_buf = ggml_backend_alloc_buffer(ctx->ggml_backend,
                                (k_size + v_size) * hp.dec_n_layers);
        char * base = (char *)ggml_backend_buffer_get_base(ctx->cross_kv_buf);

        for (int il = 0; il < hp.dec_n_layers; il++) {
            ggml_backend_tensor_alloc(ctx->cross_kv_buf, ctx->cross_kv_k[il],
                                      (void *)(base + il * (k_size + v_size)));
            ggml_backend_tensor_alloc(ctx->cross_kv_buf, ctx->cross_kv_v[il],
                                      (void *)(base + il * (k_size + v_size) + k_size));
        }

        const int n_chunks = (int)T_enc_chunks.size();
        if (n_chunks == 1) {
            // Fast path: single chunk — direct upload from CPU vectors
            for (int il = 0; il < hp.dec_n_layers; il++) {
                ggml_backend_tensor_set(ctx->cross_kv_k[il], partial_k[il][0].data(),
                                        0, partial_k[il][0].size() * sizeof(float));
                ggml_backend_tensor_set(ctx->cross_kv_v[il], partial_v[il][0].data(),
                                        0, partial_v[il][0].size() * sizeof(float));
            }
        } else {
            // Multi-chunk: scatter-copy each chunk's K/V into the final layout.
            //
            // K layout [head_dim, T_enc_total, n_heads]: for head h, all T frames are at
            //   offset h × T_enc_total × head_dim (a contiguous block of T×head_dim floats).
            //   Each chunk's head-h block: chunk_k[h × T_c × head_dim .. (h+1) × T_c × head_dim)
            //
            // V layout [T_enc_total, head_dim, n_heads]: for head h and dim d, the T frames are at
            //   h × head_dim × T_enc_total + d × T_enc_total + T_so_far (stride 1).
            //   Each chunk's (h,d) slice: chunk_v[h × head_dim × T_c + d × T_c .. + T_c)
            std::vector<float> k_full(head_dim * T_enc_total * n_heads);
            std::vector<float> v_full(T_enc_total * head_dim * n_heads);

            for (int il = 0; il < hp.dec_n_layers; il++) {
                int T_so_far = 0;
                for (int c = 0; c < n_chunks; c++) {
                    int T_c = T_enc_chunks[c];
                    const float * ck = partial_k[il][c].data();
                    const float * cv = partial_v[il][c].data();
                    for (int h = 0; h < n_heads; h++) {
                        // K: head h block in chunk = ck[h × T_c × head_dim .. ]
                        const float * ks = ck + h * T_c * head_dim;
                        float * kd = k_full.data() + h * T_enc_total * head_dim + T_so_far * head_dim;
                        memcpy(kd, ks, T_c * head_dim * sizeof(float));
                        // V: for each dim d, copy T_c consecutive frames
                        for (int d = 0; d < head_dim; d++) {
                            const float * vs = cv + h * head_dim * T_c + d * T_c;
                            float * vd = v_full.data() + h * head_dim * T_enc_total + d * T_enc_total + T_so_far;
                            memcpy(vd, vs, T_c * sizeof(float));
                        }
                    }
                    T_so_far += T_c;
                }
                ggml_backend_tensor_set(ctx->cross_kv_k[il], k_full.data(), 0, k_full.size() * sizeof(float));
                ggml_backend_tensor_set(ctx->cross_kv_v[il], v_full.data(), 0, v_full.size() * sizeof(float));
            }
        }

        t1 = ggml_time_us();
        perf.t_cross_kv_us    = t1 - t0;
        perf.mem_cross_kv_buf = ggml_backend_buffer_get_size(ctx->cross_kv_buf);
    }

    const int T_enc = T_enc_total;

    // --- Decoder prompt ---
    auto tid = [&](const std::string & s) { return voc.token_id(s); };
    const char * lang_tok = lang ? lang : "en";
    char lang_tok_str[32];
    snprintf(lang_tok_str, sizeof(lang_tok_str), "<|%s|>", lang_tok);

    std::vector<int> prompt = {
        tid("<|startofcontext|>"), tid("<|startoftranscript|>"), tid("<|emo:undefined|>"),
        tid(lang_tok_str), tid(lang_tok_str),
        tid("<|pnc|>"), tid("<|noitn|>"), tid("<|notimestamp|>"), tid("<|nodiarize|>"),
    };
    prompt.erase(std::remove_if(prompt.begin(), prompt.end(),
                                [](int t){ return t == -1; }), prompt.end());
    fprintf(stderr, "cohere: prompt               n=%d\n", (int)prompt.size());
    cohere_debug("cohere: prompt IDs: ");
    for (int id : prompt) cohere_debug("%d ", id);
    cohere_debug("\n");

    ggml_backend_buffer_clear(ctx->kv_k->buffer, 0);
    ggml_backend_buffer_clear(ctx->kv_v->buffer, 0);

    const int eos_id  = tid("<|endoftext|>");
    const int max_gen = hp.dec_max_ctx - (int)prompt.size() - 4;

    // Prompt pass
    auto logits = cohere_decode_step(ctx, T_enc, prompt.data(), (int)prompt.size(), 0);
    perf.n_dec_steps++;
    int offset = (int)prompt.size();
    fprintf(stderr, "cohere: prompt pass done      nodes=%d  build=%.1f alloc=%.1f compute=%.1f ms\n",
            perf.dec_n_nodes_prompt,
            perf.t_dec_build_us / 1e3,
            perf.t_dec_alloc_us / 1e3,
            perf.t_dec_compute_us / 1e3);

    // Greedy decode
    std::vector<int> generated;
    for (int step = 0; step < max_gen; step++) {
        const int vocab = hp.vocab_size;
        const float * last_logits = (step == 0)
            ? logits.data() + ((int)prompt.size() - 1) * vocab
            : logits.data();

        int next_tok = (int)(std::max_element(last_logits, last_logits + vocab) - last_logits);
        fprintf(stderr, "cohere: step %3d  tok=%5d  %s\n",
                step, next_tok,
                (next_tok >= 0 && next_tok < (int)voc.id_to_token.size())
                    ? voc.id_to_token[next_tok].c_str() : "?");
        if (next_tok == eos_id || next_tok < 0) break;

        generated.push_back(next_tok);
        offset++;

        logits = cohere_decode_step(ctx, T_enc, &next_tok, 1, offset - 1);
        perf.n_dec_steps++;
    }

    // Snapshot memory
    perf.mem_model_buf    = ctx->model.buf ? ggml_backend_buffer_get_size(ctx->model.buf) : 0;
    perf.mem_kv_buf       = (ctx->kv_k && ctx->kv_k->buffer)
                            ? ggml_backend_buffer_get_size(ctx->kv_k->buffer) : 0;
    perf.mem_compute_meta = ctx->compute_meta.size();
    perf.t_total_us       = ggml_time_us() - t_total_start;

    cohere_perf_print(perf, n_samples, hp.sample_rate);

    // --- Decode tokens to text ---
    std::string text;
    for (int id : generated) {
        if (id < 0 || id >= (int)voc.id_to_token.size()) continue;
        const std::string & tok = voc.id_to_token[id];
        if (tok.front() == '<' && tok.back() == '>') continue;
        std::string t = tok;
        size_t pos;
        while ((pos = t.find("\xe2\x96\x81")) != std::string::npos) t.replace(pos, 3, " ");
        text += t;
    }
    if (!text.empty() && text[0] == ' ') text = text.substr(1);

    char * result = (char *)malloc(text.size() + 1);
    memcpy(result, text.c_str(), text.size() + 1);
    return result;
}
