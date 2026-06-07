// voxcpm2_tts.cpp — VoxCPM2 TTS runtime implementation.
//
// Architecture:
//   TSLM:   28L MiniCPM-4, d=2048, 16Q/2KV heads, head_dim=128, ff=6144, LongRoPE
//   RALM:   8L  MiniCPM-4, d=2048, 16Q/2KV heads, head_dim=128, ff=6144, no RoPE
//   LocEnc: 12L bidirectional transformer, d=1024, 16h, head_dim=128, ff=4096, CLS token
//   LocDiT: 12L bidirectional transformer, d=1024, 16h, head_dim=128, ff=4096, sinusoidal time
//   FSQ:    Linear(2048->512)->tanh->round(x*9)/9->Linear(512->2048)
//   VAE:    causal transposed convolutions, Snake1d, weight-norm, SR conditioning
//
// Loading: two-pass GGUF via core_gguf::open_metadata / load_weights.
static int g_cpu_n_threads = 4;
// Matmul:  tiny ggml graphs on a shared CPU backend (g_cpu_backend) for
//          matrix-vector; manual loop (get_row_f32) for matrix-matrix prefill.
// KV cache: manual std::vector<float> per layer (CPU side).

#include "voxcpm2_tts.h"
#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/torch_rng.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

static bool vox_env_bool(const char* k) {
    const char* v = std::getenv(k);
    return v && *v && std::strcmp(v, "0") != 0;
}

// Like vox_env_bool but defaults to true (opt-out instead of opt-in).
static bool vox_env_bool_default_on(const char* k) {
    const char* v = std::getenv(k);
    if (!v || !*v)
        return true;                 // unset → on
    return std::strcmp(v, "0") != 0; // "0" → off, anything else → on
}

static double vox_now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Shared CPU backend for tiny ggml graph matmuls. Note: tried switching to
// ggml_backend_init_best (Metal/CUDA) here but the current matmul_mv_ggml
// allocates input tensors in a CPU-side mem buffer that Metal can't read
// → SIGSEGV on first kernel dispatch. The proper fix is the per-step graph
// refactor (build_locdit_graph, build_tslm_step_graph) with
// ggml_backend_tensor_set / ggml_backend_alloc_ctx_tensors — those WILL
// pick up Metal automatically once they're in place.
// ---------------------------------------------------------------------------

static ggml_backend_t g_cpu_backend = nullptr;

static ggml_backend_t get_cpu_backend() {
    if (!g_cpu_backend) {
        g_cpu_backend = ggml_backend_cpu_init();
    }
    return g_cpu_backend;
}

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct vox_hparams {
    // TSLM
    uint32_t tslm_n_layers = 28;
    uint32_t tslm_d_model = 2048;
    uint32_t tslm_n_heads = 16;
    uint32_t tslm_n_kv = 2;
    uint32_t tslm_head_dim = 128;
    uint32_t tslm_ff_dim = 6144;
    uint32_t tslm_max_pos = 32768;
    float tslm_rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;

    // RALM
    uint32_t ralm_n_layers = 8;
    uint32_t ralm_d_model = 2048;
    uint32_t ralm_n_heads = 16;
    uint32_t ralm_n_kv = 2;
    uint32_t ralm_head_dim = 128;
    uint32_t ralm_ff_dim = 6144;

    // LocEnc
    uint32_t locenc_n_layers = 12;
    uint32_t locenc_d_model = 1024;
    uint32_t locenc_n_heads = 16;
    uint32_t locenc_n_kv = 16;
    uint32_t locenc_head_dim = 128;
    uint32_t locenc_ff_dim = 4096;

    // LocDiT
    uint32_t locdit_n_layers = 12;
    uint32_t locdit_d_model = 1024;
    uint32_t locdit_n_heads = 16;
    uint32_t locdit_n_kv = 16;
    uint32_t locdit_head_dim = 128;
    uint32_t locdit_ff_dim = 4096;

    // Tokenizer
    uint32_t n_vocab = 100000;
    uint32_t audio_start_token = 0;
    uint32_t audio_end_token = 0;
    uint32_t ref_audio_start_token = 0;
    uint32_t ref_audio_end_token = 0;

    // Patch / VAE
    uint32_t patch_frames = 4;
    uint32_t patch_dim = 256;

    // VAE encoder (used for voice cloning — encodes 16 kHz mono WAV into
    // latent patches). Defaults match audio_vae_v2.py AudioVAEConfig defaults.
    uint32_t vae_enc_dim = 128;               // initial channel count
    uint32_t vae_enc_latent_dim = 64;         // == feat_dim
    uint32_t vae_enc_sample_rate = 16000;     // expected input SR
    uint32_t vae_enc_n_blocks = 4;            // number of strided blocks
    uint32_t vae_enc_rates[4] = {2, 5, 8, 8}; // per-block downsample
};

// ---------------------------------------------------------------------------
// Weight tensor structs
// ---------------------------------------------------------------------------

struct vox_lm_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct vox_enc_layer {
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct vox_vae_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct vox_weights {
    // TSLM
    ggml_tensor* tslm_token_embd = nullptr;
    ggml_tensor* tslm_output_norm = nullptr;
    // NOTE: tslm.lm_head.weight is NOT in the GGUF — omitted intentionally
    std::vector<vox_lm_layer> tslm_layers;

    // RoPE factors (LongRoPE)
    ggml_tensor* tslm_rope_short = nullptr;
    ggml_tensor* tslm_rope_long = nullptr;

    // RALM
    ggml_tensor* ralm_output_norm = nullptr;
    std::vector<vox_lm_layer> ralm_layers;

    // FSQ
    ggml_tensor* fsq_in_proj_w = nullptr;
    ggml_tensor* fsq_in_proj_b = nullptr;
    ggml_tensor* fsq_out_proj_w = nullptr;
    ggml_tensor* fsq_out_proj_b = nullptr;

    // LocEnc
    ggml_tensor* locenc_cls_token = nullptr;
    ggml_tensor* locenc_in_proj_w = nullptr; // [64, 1024] feat_dim→d_model
    ggml_tensor* locenc_in_proj_b = nullptr; // [1024]
    ggml_tensor* locenc_norm_w = nullptr;
    std::vector<vox_enc_layer> locenc_layers;

    // LocDiT
    ggml_tensor* locdit_in_proj_w = nullptr; // [64, 1024] raw latent→d_model
    ggml_tensor* locdit_in_proj_b = nullptr;
    ggml_tensor* locdit_cond_proj_w = nullptr; // [64, 1024] cond patch→d_model
    ggml_tensor* locdit_cond_proj_b = nullptr;
    ggml_tensor* locdit_time_mlp_0_w = nullptr; // [1024, 1024] sinusoidal→hidden
    ggml_tensor* locdit_time_mlp_0_b = nullptr;
    ggml_tensor* locdit_time_mlp_1_w = nullptr; // [1024, 1024] hidden→t_emb
    ggml_tensor* locdit_time_mlp_1_b = nullptr;
    ggml_tensor* locdit_dt_mlp_0_w = nullptr; // [1024, 1024] dt sinusoidal→hidden
    ggml_tensor* locdit_dt_mlp_0_b = nullptr;
    ggml_tensor* locdit_dt_mlp_1_w = nullptr; // [1024, 1024] hidden→dt_emb
    ggml_tensor* locdit_dt_mlp_1_b = nullptr;
    ggml_tensor* locdit_norm_w = nullptr;
    ggml_tensor* locdit_out_proj_w = nullptr; // [1024, 64] d_model→feat_dim
    ggml_tensor* locdit_out_proj_b = nullptr;
    std::vector<vox_enc_layer> locdit_layers;

    // Projection heads
    ggml_tensor* enc_to_lm_w = nullptr;
    ggml_tensor* enc_to_lm_b = nullptr;
    ggml_tensor* lm_to_dit_w = nullptr;
    ggml_tensor* lm_to_dit_b = nullptr;
    ggml_tensor* res_to_dit_w = nullptr;
    ggml_tensor* res_to_dit_b = nullptr;
    ggml_tensor* fusion_w = nullptr; // proj.fusion.weight [4096, 2048]
    ggml_tensor* fusion_b = nullptr; // proj.fusion.bias   [2048]

    // Stop predictor
    ggml_tensor* stop_proj_w = nullptr;
    ggml_tensor* stop_proj_b = nullptr;
    ggml_tensor* stop_head_w = nullptr; // [2048, 2] — no bias

    // VAE decoder (optional — graceful degradation if absent)
    // Weights stored in ctx->tensors under "vae.dec.*" keys
    vox_vae_conv vae_in_conv;
    vox_vae_conv vae_out_conv;
    ggml_tensor* vae_out_snake_a = nullptr;
    ggml_tensor* vae_sr_cond_w = nullptr;
    ggml_tensor* vae_sr_cond_b = nullptr;
};

// ---------------------------------------------------------------------------
// KV cache — manual CPU float storage, one vector per layer
// ---------------------------------------------------------------------------

struct vox_kv_cache {
    int n_layers = 0;
    int n_kv = 0;
    int head_dim = 0;
    int max_ctx = 0;
    int n_past = 0;
    // k_cache[l]: [max_ctx * n_kv * head_dim] row-major
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;

    void init(int layers, int kv_heads, int hd, int max_context) {
        n_layers = layers;
        n_kv = kv_heads;
        head_dim = hd;
        max_ctx = max_context;
        k_cache.assign(layers, std::vector<float>((size_t)max_context * kv_heads * hd, 0.0f));
        v_cache.assign(layers, std::vector<float>((size_t)max_context * kv_heads * hd, 0.0f));
        n_past = 0;
    }

    void reset() { n_past = 0; }
};

// ---------------------------------------------------------------------------
// Tokenizer — BPE with GPT-2 byte encoding
// ---------------------------------------------------------------------------

struct vox_tokenizer {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" -> rank
    std::vector<std::string> id_to_token;
    int32_t audio_start_token = -1;
    int32_t eos_token = -1;
};

// ---------------------------------------------------------------------------
// MT19937 + torch.randn-compatible noise live in core/torch_rng.h. We pull
// the names in here so the rest of the file reads as before.
using mt19937_state = crispasr::core::mt19937_state;
using crispasr::core::bf16_round;
using crispasr::core::fill_gaussian_noise;
using crispasr::core::fill_gaussian_noise_bf16;
using crispasr::core::mt19937_next;
using crispasr::core::mt19937_seed;
using crispasr::core::mt_uniform_torch_float;
using crispasr::core::torch_normal_fill_16;

// ---------------------------------------------------------------------------
// Context struct
// ---------------------------------------------------------------------------

struct voxcpm2_context {
    vox_hparams hp;
    vox_weights weights;
    vox_tokenizer tokenizer;
    vox_kv_cache tslm_kv;
    vox_kv_cache ralm_kv;

    // GGUF weight storage — owned here (CPU-resident when GPU is discrete)
    ggml_context* ggml_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // GPU weight mirrors.  On discrete GPUs (Vulkan, CUDA) the primary
    // weight buffer is CPU-resident so that legacy paths (tensor_data_f32,
    // matmul_mv_ggml, rms_norm_cpu, …) can dereference tensor->data.
    // Graph-build functions reference gpu_weights instead so the ggml
    // cgraph runs entirely on the GPU backend — no cross-backend copies.
    // On unified-memory backends (Metal) these stay empty; ctx->weights
    // is used everywhere.
    vox_weights gpu_weights;
    ggml_context* gpu_ggml_ctx = nullptr;
    ggml_backend_buffer_t gpu_weight_buf = nullptr;
    std::map<std::string, ggml_tensor*> gpu_tensors;
    bool has_gpu_weights = false;

    // Runtime params
    int n_threads = 4;
    int verbosity = 1;
    bool use_gpu = false;
    bool flash_attn = true;
    int inference_steps = 10;
    float cfg_value = 2.0f;
    int max_len = 2000;
    uint32_t seed = 0;

    // RNG for CFM noise generation (seeded per synthesis call)
    mt19937_state rng;

    // Helper: return gpu_weights for graph build (GPU-resident) or weights
    // for legacy paths (always CPU-accessible).
    const vox_weights& graph_weights() const { return has_gpu_weights ? gpu_weights : weights; }

    // VOXCPM2_USE_GRAPH backend pool.
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    std::vector<uint8_t> compute_meta;
    ggml_gallocr_t galloc = nullptr;

    // Cached LocDiT cgraph. LocDiT topology is constant across calls (no
    // n_past or KV cache), so we build the graph once into a dedicated
    // arena + reserve the gallocr layout once on first use; subsequent
    // CFM Euler iterations just rebind inputs and compute. Saves the
    // ~30+ ms / call cost of rebuilding the 12-layer graph + walking
    // gallocr's planner.
    std::vector<uint8_t> locdit_arena_meta;
    ggml_context* locdit_arena_ctx = nullptr;
    ggml_cgraph* locdit_gf = nullptr;
    ggml_gallocr_t locdit_galloc = nullptr;

    // Cached LocEnc cgraph. Same constant-topology trick — LocEnc takes
    // a single patch [feat_dim, P] and emits a CLS hidden state [d_enc].
    // Voice cloning hits this ~70x in build_prefill_inputs (one call per
    // reference patch), so amortising the ~250-node graph build across
    // all of them via gallocr_reserve is a real win on top of Metal.
    std::vector<uint8_t> locenc_arena_meta;
    ggml_context* locenc_arena_ctx = nullptr;
    ggml_cgraph* locenc_gf = nullptr;
    ggml_gallocr_t locenc_galloc = nullptr;

    // Pre-computed weight-norm-scaled VAE conv tensors. The legacy
    // wn_reconstruct call rebuilds these (g · v / |v|) every vae_decode
    // call for ~30 conv layers; caching once per ctx lifetime is the
    // foundation needed before the full VAE ggml graph (the graph
    // needs the resolved weights as ggml_tensor*, not std::vector).
    // Used by the legacy vae_decode path today via vae_wn_get_cached;
    // the upcoming vae_decode_graph (PLAN #96 follow-on) reads the
    // same tensors directly into ggml_conv_1d / ggml_conv_transpose_1d.
    //
    // Memory layout: legacy wn_reconstruct already emits
    // `[K, in_ch, out_ch]` for forward conv and
    // `[K, out_ch_tc, in_ch_tc]` for transposed conv — both match
    // ggml's kernel ne convention directly, so the cached values can
    // feed straight into the graph ops with no further reshape.
    std::map<std::string, std::vector<float>> vae_wn_cache;

    // Dedicated ggml arena + backend buffer holding the WN-scaled VAE
    // conv weights (and SR-cond per-bucket slices, snake1d 1/(α+1e-9)
    // tensors) as `ggml_tensor*` for the VOXCPM2_USE_GRAPH=1 VAE decode
    // graph. Built once on first use; lives for the ctx lifetime.
    // Map key is the GGUF prefix without the `.weight_g`/`.weight_v`
    // suffix (e.g. "vae.dec.layer.2.block.1"); SR-cond slices use
    // suffix `.sr_scale` / `.sr_bias`; snake1d reciprocals use
    // `.inv_alpha` appended to the alpha tensor name.
    ggml_context* vae_wn_ggml_ctx = nullptr;
    ggml_backend_buffer_t vae_wn_ggml_buf = nullptr;
    std::map<std::string, ggml_tensor*> vae_wn_ggml_tensors;

    // Cached TSLM step graphs (qwen3-style multi-bucket pattern). Each
    // bucket is topology-invariant across n_past because Lk is pinned
    // to bucket_lk and positions is passed as runtime kv_indices — the
    // K/V cache write becomes a runtime-indexed scatter instead of a
    // baked static offset. AR steps within the same bucket reuse its
    // cached gf; switching buckets pays one build + reserve.
    // Bucket sizes are powers-of-two (128, 256, 512, 1024, 2048) so
    // small/typical inputs ("Hello world" zero-shot < 20 positions,
    // jfk.wav clone ~80) keep the smallest bucket and pay the least
    // attention overhead; longer prefills upgrade as needed.
    struct TslmBucket {
        int lk = 0;
        std::vector<uint8_t> arena_meta;
        ggml_context* arena_ctx = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_gallocr_t galloc = nullptr;
    };
    std::array<TslmBucket, 5> tslm_buckets;

    // Persistent KV cache as a backend tensor for VOXCPM2_USE_GRAPH=1 TSLM
    // step. Sized to match the legacy tslm_kv (max_ctx × n_kv × head_dim ×
    // n_layers). Populated lazily from tslm_kv before the AR loop's first
    // graph step, then read/written directly by the step graph.
    ggml_context* tslm_kv_ctx = nullptr;
    ggml_backend_buffer_t tslm_kv_buf = nullptr;
    ggml_tensor* tslm_kv_k = nullptr;
    ggml_tensor* tslm_kv_v = nullptr;
    int tslm_kv_max_ctx = 0;
    bool tslm_kv_synced = false; // true once tslm_kv (CPU vector) has been
                                 // copied into the backend tensor for this
                                 // synthesis call.
};

// Stream struct
struct voxcpm2_stream {
    voxcpm2_context* ctx = nullptr;
    std::vector<float> all_pcm;
    std::vector<float> chunk_buf;
    int chunk_offset = 0;
    bool done = false;
};

// ---------------------------------------------------------------------------
// get_row_f32: copy one row from any weight tensor type to float
// Handles F32, F16, Q4_K (and any type with a to_float trait) via
// ggml_get_type_traits()->to_float.
// ---------------------------------------------------------------------------

static void get_row_f32(const ggml_tensor* t, int row, float* out) {
    const int cols = (int)t->ne[0];
    if (t->type == GGML_TYPE_F32) {
        const float* src = (const float*)((const char*)t->data + (size_t)row * t->nb[1]);
        std::memcpy(out, src, (size_t)cols * sizeof(float));
        return;
    }
    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (tt && tt->to_float) {
        const void* src = (const char*)t->data + (size_t)row * t->nb[1];
        tt->to_float(src, out, cols);
        return;
    }
    std::memset(out, 0, (size_t)cols * sizeof(float));
}

// tensor_data_f32: for tensors guaranteed to be F32 (norms, biases).
static const float* tensor_data_f32(const ggml_tensor* t) {
    return (const float*)t->data;
}

// ---------------------------------------------------------------------------
// matmul_mv_ggml: W[rows,cols] x v[cols] -> out[rows] via tiny ggml graph.
// Uses ggml_mul_mat, which dispatches to SIMD / Q4_K native dot products.
// ---------------------------------------------------------------------------

static void matmul_mv_ggml(ggml_backend_t cpu_be, ggml_tensor* W, const float* v, int cols, float* out, int rows) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/(size_t)2 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ggml_context* tmp_ctx = ggml_init(ip);

    // Input vector as a 2-D column: [cols, 1]
    ggml_tensor* v_t = ggml_new_tensor_2d(tmp_ctx, GGML_TYPE_F32, cols, 1);
    std::memcpy(v_t->data, v, (size_t)cols * sizeof(float));

    // ggml_mul_mat(a, b): a[ne0, ne1] x b[ne0, 1] -> result[ne1, 1]
    // W tensor from GGUF: ne[0] = inner dim, ne[1] = outer dim
    // We need W->ne[0] == cols for the matmul to work
    if ((int)W->ne[0] != cols) {
        fprintf(stderr, "voxcpm2: matmul_mv_ggml dim mismatch: W[%lld,%lld] x v[%d] (expected W->ne[0]==%d)\n",
                (long long)W->ne[0], (long long)W->ne[1], cols, cols);
    }
    ggml_tensor* result = ggml_mul_mat(tmp_ctx, W, v_t);

    ggml_cgraph* gf = ggml_new_graph(tmp_ctx);
    ggml_build_forward_expand(gf, result);

    if (ggml_backend_is_cpu(cpu_be)) {
        ggml_backend_cpu_set_n_threads(cpu_be, g_cpu_n_threads);
    }
    ggml_backend_graph_compute(cpu_be, gf);

    std::memcpy(out, result->data, (size_t)rows * sizeof(float));
    ggml_free(tmp_ctx);
}

static void matmul_mv(ggml_backend_t cpu_be, ggml_tensor* W, const float* v, int cols, float* out, int rows) {
    matmul_mv_ggml(cpu_be, W, v, cols, out, rows);
}

static void matmul_mv_bias(ggml_backend_t cpu_be, ggml_tensor* W, ggml_tensor* b_t, const float* v, int cols,
                           float* out, int rows) {
    matmul_mv_ggml(cpu_be, W, v, cols, out, rows);
    const float* b = tensor_data_f32(b_t);
    for (int i = 0; i < rows; i++)
        out[i] += b[i];
}

// matmul_mm: W[rows,cols] x X[cols,T] -> Y[T,rows]  (manual loop, for prefill)
static void matmul_mm(ggml_tensor* W, const float* X, int cols, int rows, int T, float* Y) {
    std::vector<float> w_row((size_t)cols);
    for (int r = 0; r < rows; r++) {
        get_row_f32(W, r, w_row.data());
        for (int t = 0; t < T; t++) {
            float acc = 0.0f;
            const float* xc = X + (size_t)t * cols;
            for (int c = 0; c < cols; c++)
                acc += w_row[c] * xc[c];
            Y[(size_t)t * rows + r] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// MT19937 + torch.randn (F32 and BF16) noise live in core/torch_rng.h.
// The using-declarations earlier in this file pull them into scope.
// ---------------------------------------------------------------------------
// RMS Norm (CPU, in-place-safe: x and y may differ)
// ---------------------------------------------------------------------------

static void rms_norm_cpu(const float* x, const float* w, float* y, int d, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++)
        ss += x[i] * x[i];
    ss = ss / (float)d + eps;
    float inv = 1.0f / std::sqrt(ss);
    for (int i = 0; i < d; i++)
        y[i] = x[i] * inv * w[i];
}

// ---------------------------------------------------------------------------
// SwiGLU FFN: out = W_down @ (silu(W_gate @ x) * W_up @ x)
// ---------------------------------------------------------------------------

static void swiglu_ffn_cpu(ggml_backend_t cpu_be, ggml_tensor* gate_w, ggml_tensor* up_w, ggml_tensor* down_w,
                           const float* x, int d_in, int d_ff, int d_out, float* y) {
    std::vector<float> gate(d_ff), up(d_ff), h(d_ff);
    matmul_mv(cpu_be, gate_w, x, d_in, gate.data(), d_ff);
    matmul_mv(cpu_be, up_w, x, d_in, up.data(), d_ff);
    for (int i = 0; i < d_ff; i++) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + std::exp(-g));
        h[i] = g * sig * up[i];
    }
    matmul_mv(cpu_be, down_w, h.data(), d_ff, y, d_out);
}

// ---------------------------------------------------------------------------
// NEOX RoPE (LongRoPE baseline: no freq scaling, standard complex rotation)
// Operates on interleaved layout: pairs (i, i+head_dim/2) within each head.
// ---------------------------------------------------------------------------

static void rope_apply_cpu(float* qk, int head_dim, int n_heads, int pos, float theta, int /*n_ctx_orig*/,
                           const float* short_factors = nullptr) {
    for (int h = 0; h < n_heads; h++) {
        float* vec = qk + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float inv_freq = 1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * inv_freq;
            if (short_factors)
                angle /= short_factors[i];
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float x0 = vec[i];
            float x1 = vec[i + head_dim / 2];
            vec[i] = x0 * cos_a - x1 * sin_a;
            vec[i + head_dim / 2] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal attention step — single new token, writes into KV cache, reads full
// history [0, n_past+1), returns output [n_q * head_dim].
// ---------------------------------------------------------------------------

static void causal_attn_step(const float* q_in,  // [n_q  * hd]
                             const float* k_new, // [n_kv * hd]
                             const float* v_new, // [n_kv * hd]
                             float* out,         // [n_q  * hd]
                             vox_kv_cache& cache, int layer, int n_q, int n_kv, int hd, float attn_scale) {
    int n_past = cache.n_past;
    int seq_len = n_past + 1;
    int grp = n_q / n_kv;

    // Write new K/V into cache at position n_past
    {
        float* kd = cache.k_cache[layer].data() + (size_t)n_past * n_kv * hd;
        float* vd = cache.v_cache[layer].data() + (size_t)n_past * n_kv * hd;
        std::memcpy(kd, k_new, (size_t)n_kv * hd * sizeof(float));
        std::memcpy(vd, v_new, (size_t)n_kv * hd * sizeof(float));
    }

    std::vector<float> scores(seq_len), attn_w(seq_len);

    for (int qh = 0; qh < n_q; qh++) {
        const float* q = q_in + qh * hd;
        int kvh = qh / grp;

        // Dot products Q.K^T over cached history
        for (int t = 0; t < seq_len; t++) {
            const float* k_t = cache.k_cache[layer].data() + (size_t)t * n_kv * hd + kvh * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; d++)
                dot += q[d] * k_t[d];
            scores[t] = dot * attn_scale;
        }

        // Stable softmax
        float max_s = scores[0];
        for (int t = 1; t < seq_len; t++)
            if (scores[t] > max_s)
                max_s = scores[t];
        float sum_e = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            attn_w[t] = std::exp(scores[t] - max_s);
            sum_e += attn_w[t];
        }
        for (int t = 0; t < seq_len; t++)
            attn_w[t] /= sum_e;

        // Weighted sum over V
        float* o = out + qh * hd;
        std::fill(o, o + hd, 0.0f);
        for (int t = 0; t < seq_len; t++) {
            const float* v_t = cache.v_cache[layer].data() + (size_t)t * n_kv * hd + kvh * hd;
            for (int d = 0; d < hd; d++)
                o[d] += attn_w[t] * v_t[d];
        }
    }
}

// ---------------------------------------------------------------------------
// LongRoPE positional embedding for bidirectional attention.
// Applies in-place to Q[T * n_heads * hd] and K[T * n_kv * hd].
// short_factors: [hd/2] factors; theta: base frequency; scaling: overall scale.
// Formula: angle = pos / (short_factor[i] * theta^(2i/hd))
// Rotation: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
//   where x0 = vec[i], x1 = vec[i + hd/2]  (rotate_half convention)
// ---------------------------------------------------------------------------

static void longrope_apply_bidir(float* Q, float* K, int T, int n_q, int n_kv, int hd, float theta,
                                 const float* short_factors) {
    int half_hd = hd / 2;
    for (int pos = 0; pos < T; pos++) {
        // Apply to Q heads
        for (int h = 0; h < n_q; h++) {
            float* vec = Q + (size_t)pos * n_q * hd + (size_t)h * hd;
            for (int i = 0; i < half_hd; i++) {
                float inv_freq = 1.0f / std::pow(theta, (float)(2 * i) / (float)hd);
                float angle = (float)pos * inv_freq / short_factors[i];
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float x0 = vec[i];
                float x1 = vec[i + half_hd];
                vec[i] = x0 * cos_a - x1 * sin_a;
                vec[i + half_hd] = x0 * sin_a + x1 * cos_a;
            }
        }
        // Apply to K heads
        for (int h = 0; h < n_kv; h++) {
            float* vec = K + (size_t)pos * n_kv * hd + (size_t)h * hd;
            for (int i = 0; i < half_hd; i++) {
                float inv_freq = 1.0f / std::pow(theta, (float)(2 * i) / (float)hd);
                float angle = (float)pos * inv_freq / short_factors[i];
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float x0 = vec[i];
                float x1 = vec[i + half_hd];
                vec[i] = x0 * cos_a - x1 * sin_a;
                vec[i + half_hd] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bidirectional (full) attention for LocEnc / LocDiT — no mask
// x_in: [T * d] row-major, out: [T * d] row-major
// When rope_factors != nullptr, applies LongRoPE to Q/K before attention.
// ---------------------------------------------------------------------------

static void bidir_attn_full(const float* x_in, int T, int d, ggml_tensor* q_w, ggml_tensor* k_w, ggml_tensor* v_w,
                            ggml_tensor* o_w, int n_q, int n_kv, int hd, float attn_scale, ggml_backend_t cpu_be,
                            float* out, const float* rope_factors = nullptr, float rope_theta = 10000.0f) {
    int grp = n_q / n_kv;

    std::vector<float> Q((size_t)T * n_q * hd);
    std::vector<float> K((size_t)T * n_kv * hd);
    std::vector<float> V((size_t)T * n_kv * hd);

    for (int t = 0; t < T; t++) {
        const float* xt = x_in + (size_t)t * d;
        matmul_mv(cpu_be, q_w, xt, d, Q.data() + (size_t)t * n_q * hd, n_q * hd);
        matmul_mv(cpu_be, k_w, xt, d, K.data() + (size_t)t * n_kv * hd, n_kv * hd);
        matmul_mv(cpu_be, v_w, xt, d, V.data() + (size_t)t * n_kv * hd, n_kv * hd);
    }

    // Apply LongRoPE if factors provided
    if (rope_factors) {
        longrope_apply_bidir(Q.data(), K.data(), T, n_q, n_kv, hd, rope_theta, rope_factors);
    }

    std::vector<float> attn_out((size_t)T * n_q * hd, 0.0f);
    std::vector<float> scores(T), aw(T);

    for (int t = 0; t < T; t++) {
        for (int qh = 0; qh < n_q; qh++) {
            const float* q = Q.data() + (size_t)t * n_q * hd + qh * hd;
            int kvh = qh / grp;

            for (int s = 0; s < T; s++) {
                const float* ks = K.data() + (size_t)s * n_kv * hd + kvh * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; i++)
                    dot += q[i] * ks[i];
                scores[s] = dot * attn_scale;
            }

            float max_s = scores[0];
            for (int s = 1; s < T; s++)
                if (scores[s] > max_s)
                    max_s = scores[s];
            float sum_e = 0.0f;
            for (int s = 0; s < T; s++) {
                aw[s] = std::exp(scores[s] - max_s);
                sum_e += aw[s];
            }
            for (int s = 0; s < T; s++)
                aw[s] /= sum_e;

            float* ao = attn_out.data() + (size_t)t * n_q * hd + qh * hd;
            for (int s = 0; s < T; s++) {
                const float* vs = V.data() + (size_t)s * n_kv * hd + kvh * hd;
                for (int i = 0; i < hd; i++)
                    ao[i] += aw[s] * vs[i];
            }
        }
    }

    // Output projection: attn_out[T, n_q*hd] x o_w -> out[T, d]
    for (int t = 0; t < T; t++) {
        matmul_mv(cpu_be, o_w, attn_out.data() + (size_t)t * n_q * hd, n_q * hd, out + (size_t)t * d, d);
    }
}

// ---------------------------------------------------------------------------
// TSLM layer — single-token causal decode with KV cache
// ---------------------------------------------------------------------------

static void tslm_layer_step(voxcpm2_context* ctx, int layer, float* hidden, int pos, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_lm_layer& L = ctx->weights.tslm_layers[layer];
    int d = (int)hp.tslm_d_model;
    int n_q = (int)hp.tslm_n_heads;
    int n_kv = (int)hp.tslm_n_kv;
    int hd = (int)hp.tslm_head_dim;
    float eps = hp.rms_norm_eps;
    float attn_scale = 1.0f / std::sqrt((float)hd);

    std::vector<float> normed(d), q(n_q * hd), k(n_kv * hd), v(n_kv * hd), attn_out(n_q * hd), proj_out(d), ffn_out(d);

    rms_norm_cpu(hidden, tensor_data_f32(L.attn_norm_w), normed.data(), d, eps);

    matmul_mv(cpu_be, L.attn_q_w, normed.data(), d, q.data(), n_q * hd);
    matmul_mv(cpu_be, L.attn_k_w, normed.data(), d, k.data(), n_kv * hd);
    matmul_mv(cpu_be, L.attn_v_w, normed.data(), d, v.data(), n_kv * hd);

    const float* tslm_rope_sf = ctx->weights.tslm_rope_short ? tensor_data_f32(ctx->weights.tslm_rope_short) : nullptr;
    rope_apply_cpu(q.data(), hd, n_q, pos, hp.tslm_rope_theta, (int)hp.tslm_max_pos, tslm_rope_sf);
    rope_apply_cpu(k.data(), hd, n_kv, pos, hp.tslm_rope_theta, (int)hp.tslm_max_pos, tslm_rope_sf);

    causal_attn_step(q.data(), k.data(), v.data(), attn_out.data(), ctx->tslm_kv, layer, n_q, n_kv, hd, attn_scale);

    matmul_mv(cpu_be, L.attn_o_w, attn_out.data(), n_q * hd, proj_out.data(), d);
    for (int i = 0; i < d; i++)
        hidden[i] += proj_out[i];

    rms_norm_cpu(hidden, tensor_data_f32(L.ffn_norm_w), normed.data(), d, eps);
    swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data(), d, (int)hp.tslm_ff_dim, d,
                   ffn_out.data());
    for (int i = 0; i < d; i++)
        hidden[i] += ffn_out[i];
}

// ---------------------------------------------------------------------------
// RALM layer — causal decode, no RoPE
// ---------------------------------------------------------------------------

static void ralm_layer_step(voxcpm2_context* ctx, int layer, float* hidden, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_lm_layer& L = ctx->weights.ralm_layers[layer];
    int d = (int)hp.ralm_d_model;
    int n_q = (int)hp.ralm_n_heads;
    int n_kv = (int)hp.ralm_n_kv;
    int hd = (int)hp.ralm_head_dim;
    float eps = hp.rms_norm_eps;
    float attn_scale = 1.0f / std::sqrt((float)hd);

    std::vector<float> normed(d), q(n_q * hd), k(n_kv * hd), v(n_kv * hd), attn_out(n_q * hd), proj_out(d), ffn_out(d);

    rms_norm_cpu(hidden, tensor_data_f32(L.attn_norm_w), normed.data(), d, eps);

    matmul_mv(cpu_be, L.attn_q_w, normed.data(), d, q.data(), n_q * hd);
    matmul_mv(cpu_be, L.attn_k_w, normed.data(), d, k.data(), n_kv * hd);
    matmul_mv(cpu_be, L.attn_v_w, normed.data(), d, v.data(), n_kv * hd);

    // No RoPE for RALM
    causal_attn_step(q.data(), k.data(), v.data(), attn_out.data(), ctx->ralm_kv, layer, n_q, n_kv, hd, attn_scale);

    matmul_mv(cpu_be, L.attn_o_w, attn_out.data(), n_q * hd, proj_out.data(), d);
    for (int i = 0; i < d; i++)
        hidden[i] += proj_out[i];

    rms_norm_cpu(hidden, tensor_data_f32(L.ffn_norm_w), normed.data(), d, eps);
    swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data(), d, (int)hp.ralm_ff_dim, d,
                   ffn_out.data());
    for (int i = 0; i < d; i++)
        hidden[i] += ffn_out[i];
}

// ---------------------------------------------------------------------------
// VOXCPM2_USE_GRAPH=1 TSLM step path — backend-resident KV cache + a single
// 28-layer cgraph per AR-step TSLM call.
//
// Layout: backend tensor ne = (head_dim, max_ctx, n_kv, n_layers) — the
// qwen3-style layout that core_attn::kv_self_attn expects. The legacy
// vox_kv_cache stores (hd, n_kv, max_ctx, n_layers) on the host (different
// pos↔kvh order), so `sync_tslm_kv_cpu_to_backend` walks both layouts and
// transposes pos/kvh while copying. Called once per synthesis, after TSLM
// prefill has populated the legacy cache and before the first AR-step
// graph call.
// ---------------------------------------------------------------------------

static bool init_tslm_kv_backend(voxcpm2_context* ctx) {
    if (ctx->tslm_kv_k) {
        return true;
    }
    const vox_hparams& hp = ctx->hp;
    const int hd = (int)hp.tslm_head_dim;
    const int n_kv = (int)hp.tslm_n_kv;
    const int n_lay = (int)hp.tslm_n_layers;
    const int max_ctx = ctx->tslm_kv.max_ctx > 0 ? ctx->tslm_kv.max_ctx : 4096;

    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, /*no_alloc=*/true};
    ctx->tslm_kv_ctx = ggml_init(kp);
    if (!ctx->tslm_kv_ctx) {
        return false;
    }
    ctx->tslm_kv_k = ggml_new_tensor_4d(ctx->tslm_kv_ctx, GGML_TYPE_F32, hd, max_ctx, n_kv, n_lay);
    ctx->tslm_kv_v = ggml_new_tensor_4d(ctx->tslm_kv_ctx, GGML_TYPE_F32, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->tslm_kv_k, "tslm_kv_k");
    ggml_set_name(ctx->tslm_kv_v, "tslm_kv_v");
    const size_t kb = ggml_nbytes(ctx->tslm_kv_k);
    const size_t vb = ggml_nbytes(ctx->tslm_kv_v);
    ctx->tslm_kv_buf = ggml_backend_alloc_buffer(ctx->backend, kb + vb);
    if (!ctx->tslm_kv_buf) {
        fprintf(stderr, "voxcpm2: failed to alloc tslm kv backend buffer (%zu bytes)\n", kb + vb);
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->tslm_kv_buf);
    ggml_backend_tensor_alloc(ctx->tslm_kv_buf, ctx->tslm_kv_k, base);
    ggml_backend_tensor_alloc(ctx->tslm_kv_buf, ctx->tslm_kv_v, base + kb);
    ggml_backend_buffer_clear(ctx->tslm_kv_buf, 0);
    ctx->tslm_kv_max_ctx = max_ctx;
    return true;
}

// Copy legacy CPU KV cache (vox_kv_cache, pos-major) into the backend
// tensor (kvh-major). Writes one (max_ctx × hd) layer slice per call to
// ggml_backend_tensor_set, transposing pos↔kvh into a small staging buffer.
static void sync_tslm_kv_cpu_to_backend(voxcpm2_context* ctx) {
    const vox_hparams& hp = ctx->hp;
    const int hd = (int)hp.tslm_head_dim;
    const int n_kv = (int)hp.tslm_n_kv;
    const int n_lay = (int)hp.tslm_n_layers;
    const int n_past = ctx->tslm_kv.n_past;
    const int max_ctx = ctx->tslm_kv_max_ctx;
    if (n_past <= 0 || !ctx->tslm_kv_k || !ctx->tslm_kv_v) {
        return;
    }
    std::vector<float> stage((size_t)max_ctx * n_kv * hd, 0.0f);
    const size_t layer_bytes = (size_t)max_ctx * n_kv * hd * sizeof(float);
    for (int layer = 0; layer < n_lay; layer++) {
        // K
        std::fill(stage.begin(), stage.end(), 0.0f);
        for (int kvh = 0; kvh < n_kv; kvh++) {
            for (int pos = 0; pos < n_past; pos++) {
                float* dst = stage.data() + (size_t)(kvh * max_ctx + pos) * hd;
                const float* src = ctx->tslm_kv.k_cache[layer].data() + (size_t)pos * n_kv * hd + (size_t)kvh * hd;
                std::memcpy(dst, src, (size_t)hd * sizeof(float));
            }
        }
        ggml_backend_tensor_set(ctx->tslm_kv_k, stage.data(), (size_t)layer * layer_bytes, layer_bytes);
        // V
        std::fill(stage.begin(), stage.end(), 0.0f);
        for (int kvh = 0; kvh < n_kv; kvh++) {
            for (int pos = 0; pos < n_past; pos++) {
                float* dst = stage.data() + (size_t)(kvh * max_ctx + pos) * hd;
                const float* src = ctx->tslm_kv.v_cache[layer].data() + (size_t)pos * n_kv * hd + (size_t)kvh * hd;
                std::memcpy(dst, src, (size_t)hd * sizeof(float));
            }
        }
        ggml_backend_tensor_set(ctx->tslm_kv_v, stage.data(), (size_t)layer * layer_bytes, layer_bytes);
    }
}

// Build the per-step TSLM cgraph (all 28 layers, T=1). Reuses
// core_attn::kv_self_attn (NEOX RoPE with LongRoPE freq_factors, GQA
// expansion, flash-attn) and core_ffn::swiglu. The graph writes the new
// (K, V) into ctx->tslm_kv_{k,v} at position n_past and reads the full
// [0, n_past+1) history. The final RMSNorm + output_norm scale are
// folded into the graph so the caller gets the already-normed hidden
// state — same value that the legacy path produces after its post-loop
// rms_norm_cpu(... tslm_output_norm ...).
static ggml_cgraph* build_tslm_step_graph(voxcpm2_context* ctx, int n_past, int fixed_kv_len = 0,
                                          ggml_context* arena_ctx = nullptr) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W = ctx->graph_weights();
    const int d = (int)hp.tslm_d_model;
    const int n_q = (int)hp.tslm_n_heads;
    const int n_kv = (int)hp.tslm_n_kv;
    const int hd = (int)hp.tslm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = 1;
    const int Lk = fixed_kv_len > 0 ? fixed_kv_len : (n_past + T);

    // arena_ctx supplied → graph metadata persists across calls
    // (qwen3-style bucket pattern); otherwise reuse compute_meta for a
    // one-shot dynamic build.
    ggml_context* ctx0 = arena_ctx;
    if (!ctx0) {
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), /*no_alloc=*/true};
        ctx0 = ggml_init(ip);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* hidden_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(hidden_in, "hidden_in");
    ggml_set_input(hidden_in);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Bucketed (fixed_kv_len > 0) → causal_mask is required to hide the
    // unwritten tail [n_past+1, Lk). Dynamic (fixed_kv_len == 0) → Lk
    // tightly tracks n_past+T and the tail doesn't exist, so no mask.
    ggml_tensor* causal_mask = nullptr;
    if (fixed_kv_len > 0) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.tslm_max_pos,
        /*rope_theta*/ hp.tslm_rope_theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
        /*rope_type*/ GGML_ROPE_TYPE_NEOX,
        /*n_rot*/ 0,
        /*v_rms_norm*/ false,
        /*rope_freq_factors*/ W.tslm_rope_short,
    };

    // Bucketed path: K/V scatter via ggml_set_rows keyed on `positions`
    // (runtime-indexed) so the graph topology is invariant across n_past
    // — same trick as qwen3_tts' QWEN3_TTS_O15. Dynamic path: K/V scatter
    // baked into the graph as a static offset using n_past.
    ggml_tensor* eff_kv_indices = (fixed_kv_len > 0) ? positions : nullptr;
    // n_past=0 in bucketed mode keeps the read view's starting offset at
    // layer base; we still read all Lk slots and rely on causal_mask to
    // suppress unwritten ones.
    const int eff_n_past = (fixed_kv_len > 0) ? 0 : n_past;

    ggml_tensor* cur = hidden_in;
    for (uint32_t il = 0; il < hp.tslm_n_layers; il++) {
        const vox_lm_layer& L = W.tslm_layers[il];
        ggml_tensor* residual = cur;

        // Attention block (RMSNorm × scale → kv_self_attn → residual).
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, L.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, L.attn_q_w, L.attn_k_w, L.attn_v_w, L.attn_o_w,
                                                    /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask,
                                                    ctx->tslm_kv_k, ctx->tslm_kv_v, (int)il, eff_n_past, kvp,
                                                    /*qkv_w=*/nullptr, /*fixed_kv_len=*/Lk,
                                                    /*kv_indices=*/eff_kv_indices);
        cur = ggml_add(ctx0, residual, attn);

        // FFN block (RMSNorm × scale → SwiGLU → residual).
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, L.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final RMSNorm × tslm_output_norm. AR loop's legacy path applies this
    // after the per-layer loop; folding it into the graph keeps wrapper
    // arithmetic out of the hot path.
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, W.tslm_output_norm);
    ggml_set_name(cur, "hidden_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!arena_ctx) {
        ggml_free(ctx0);
    }
    return gf;
}

// Bucket policy: powers-of-two from 128 to 2048. Smallest fitting
// bucket wins — short prompts pay only the 128-bucket attn cost,
// longer ones upgrade as needed. Buckets are built lazily on first
// hit (most synths touch one or two).
static constexpr int kTslmBuckets[] = {128, 256, 512, 1024, 2048};
static constexpr int kTslmNumBuckets = (int)(sizeof(kTslmBuckets) / sizeof(kTslmBuckets[0]));

static int tslm_pick_bucket(int needed_lk) {
    for (int i = 0; i < kTslmNumBuckets; i++) {
        if (kTslmBuckets[i] >= needed_lk) {
            return i;
        }
    }
    return -1;
}

// Build / fetch the cached TSLM step cgraph for the given bucket index.
// Topology is `n_past`-invariant: K/V scatter index + causal mask are
// runtime inputs (qwen3 LK_BUCKET pattern). One bucket covers all
// AR-step n_past values up to its Lk - 1; needed_lk past the largest
// bucket falls back to the dynamic graph in the caller.
static ggml_cgraph* get_or_build_tslm_step_graph(voxcpm2_context* ctx, int bucket_idx) {
    if (bucket_idx < 0 || bucket_idx >= kTslmNumBuckets) {
        return nullptr;
    }
    auto& bk = ctx->tslm_buckets[bucket_idx];
    if (bk.gf) {
        return bk.gf;
    }
    if (!ctx->backend || !ctx->tslm_kv_k) {
        return nullptr;
    }
    const int bucket_lk = kTslmBuckets[bucket_idx];
    bk.lk = bucket_lk;
    bk.arena_meta.assign(ctx->compute_meta.size(), 0);
    ggml_init_params ip = {bk.arena_meta.size(), bk.arena_meta.data(), /*no_alloc=*/true};
    bk.arena_ctx = ggml_init(ip);
    if (!bk.arena_ctx) {
        bk.arena_meta.clear();
        return nullptr;
    }
    bk.gf = build_tslm_step_graph(ctx, /*n_past=*/0, /*fixed_kv_len=*/bucket_lk, bk.arena_ctx);
    if (!bk.gf) {
        ggml_free(bk.arena_ctx);
        bk.arena_ctx = nullptr;
        bk.arena_meta.clear();
        return nullptr;
    }
    bk.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!bk.galloc || !ggml_gallocr_reserve(bk.galloc, bk.gf)) {
        if (bk.galloc) {
            ggml_gallocr_free(bk.galloc);
            bk.galloc = nullptr;
        }
        ggml_free(bk.arena_ctx);
        bk.arena_ctx = nullptr;
        bk.gf = nullptr;
        bk.arena_meta.clear();
        return nullptr;
    }
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: built tslm step bucket Lk=%d\n", bucket_lk);
    }
    return bk.gf;
}

// Run one TSLM step through the graph. Caller must have ensured
// init_tslm_kv_backend + sync_tslm_kv_cpu_to_backend completed before
// the first invocation. Picks the smallest fitting bucket; falls
// through to a one-shot dynamic build when n_past+1 outgrows the
// largest bucket. Per-call cost in the bucketed path: tensor_set
// (hidden_in + positions + causal_mask) + ggml_backend_graph_compute
// + tensor_get. Bigger bucket trades graph-build savings for wasted
// attention work on the masked tail — flash-attn computes Q·K^T over
// the full Lk even when the tail is -inf masked, so 128 is meaningfully
// cheaper than 2048 for short prompts; multi-bucket lets us pick.
static std::vector<float> tslm_step_graph(voxcpm2_context* ctx, const float* hidden_in, int pos) {
    const int d = (int)ctx->hp.tslm_d_model;
    int32_t pos_i = pos;

    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    int Lk = 0;
    bool bucketed = false;
    const int needed_lk = pos + 1;
    const int bucket_idx = tslm_pick_bucket(needed_lk);
    if (bucket_idx >= 0) {
        gf = get_or_build_tslm_step_graph(ctx, bucket_idx);
        if (gf) {
            galloc = ctx->tslm_buckets[bucket_idx].galloc;
            Lk = kTslmBuckets[bucket_idx];
            bucketed = true;
        }
    }
    if (!gf) {
        gf = build_tslm_step_graph(ctx, pos);
        galloc = ctx->galloc;
        Lk = pos + 1;
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "voxcpm2: tslm_step gallocr alloc failed\n");
        return std::vector<float>(d, 0.0f);
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "hidden_in"), hidden_in, 0, (size_t)d * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), &pos_i, 0, sizeof(int32_t));
    if (bucketed) {
        // Mask: shape (Lk, 1). 0 for k <= pos (visible written slots);
        // -inf for k > pos (unwritten tail). F16 to match the helper's
        // declared mask type.
        std::vector<ggml_fp16_t> mask((size_t)Lk);
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int k = 0; k < Lk; k++) {
            mask[k] = (k <= pos) ? z : ninf;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, g_cpu_n_threads);
    }
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxcpm2: tslm_step graph compute failed\n");
        return std::vector<float>(d, 0.0f);
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "hidden_out");
    std::vector<float> result(d);
    ggml_backend_tensor_get(out, result.data(), 0, (size_t)d * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// TSLM prefill — run all text tokens through, filling KV cache
// Returns last hidden state [d_model] (pre-output-norm)
// ---------------------------------------------------------------------------

static std::vector<float> tslm_prefill(voxcpm2_context* ctx, const std::vector<int32_t>& token_ids,
                                       ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.tslm_d_model;
    int T = (int)token_ids.size();
    ctx->tslm_kv.reset();

    std::vector<float> hidden(d);

    for (int t = 0; t < T; t++) {
        int id = token_ids[t];
        if (id < 0 || id >= (int)hp.n_vocab)
            id = 0;
        get_row_f32(ctx->weights.tslm_token_embd, id, hidden.data());

        for (int l = 0; l < (int)hp.tslm_n_layers; l++) {
            tslm_layer_step(ctx, l, hidden.data(), t, cpu_be);
        }
        ctx->tslm_kv.n_past = t + 1;
    }

    return hidden;
}

// ---------------------------------------------------------------------------
// TSLM prefill (instrumented) — saves per-position outputs + per-layer hooks
// all_positions: if non-null, appends each position's hidden [d] (after all layers)
// layer_hooks: map from layer index → buffer; saves first N positions of that layer
// ---------------------------------------------------------------------------

struct tslm_prefill_hooks {
    std::vector<float>* all_positions = nullptr;  // [T * d] row-major
    int layer0_capture = -1;                      // layer index for "layer 0" hook
    std::vector<float>* layer0_out = nullptr;     // [N * d]
    int layer_last_capture = -1;                  // layer index for "last layer" hook
    std::vector<float>* layer_last_out = nullptr; // [N * d]
    int max_capture_positions = 8;                // capture first N positions only
};

static std::vector<float> tslm_prefill_ex(voxcpm2_context* ctx, const std::vector<int32_t>& token_ids,
                                          ggml_backend_t cpu_be, const tslm_prefill_hooks& hooks) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.tslm_d_model;
    int T = (int)token_ids.size();
    int n_layers = (int)hp.tslm_n_layers;
    ctx->tslm_kv.reset();

    std::vector<float> hidden(d);

    for (int t = 0; t < T; t++) {
        int id = token_ids[t];
        if (id < 0 || id >= (int)hp.n_vocab)
            id = 0;
        get_row_f32(ctx->weights.tslm_token_embd, id, hidden.data());

        for (int l = 0; l < n_layers; l++) {
            tslm_layer_step(ctx, l, hidden.data(), t, cpu_be);

            // Per-layer hooks (capture first N positions)
            if (t < hooks.max_capture_positions) {
                if (l == hooks.layer0_capture && hooks.layer0_out) {
                    hooks.layer0_out->insert(hooks.layer0_out->end(), hidden.data(), hidden.data() + d);
                }
                if (l == hooks.layer_last_capture && hooks.layer_last_out) {
                    hooks.layer_last_out->insert(hooks.layer_last_out->end(), hidden.data(), hidden.data() + d);
                }
            }
        }
        ctx->tslm_kv.n_past = t + 1;

        // Save all-positions output (after all layers, before output norm)
        if (hooks.all_positions && t < hooks.max_capture_positions) {
            hooks.all_positions->insert(hooks.all_positions->end(), hidden.data(), hidden.data() + d);
        }
    }

    return hidden;
}

// Variant that takes pre-computed per-position embeddings instead of token IDs.
// Used by the voice-cloning prefill where the input is a mix of text embeddings
// and LocEnc-projected ref audio features (Python:
//   combined_embed = text_mask * embed_tokens(tokens) + audio_mask * enc_to_lm_proj(feat_encoder(feats))
// ).
// `embeds` is [T * d] row-major. Captures all positions when hooks request it.
static std::vector<float> tslm_prefill_from_embeds(voxcpm2_context* ctx, const float* embeds, int T,
                                                   ggml_backend_t cpu_be, const tslm_prefill_hooks& hooks) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.tslm_d_model;
    int n_layers = (int)hp.tslm_n_layers;
    ctx->tslm_kv.reset();

    std::vector<float> hidden(d);
    for (int t = 0; t < T; t++) {
        std::memcpy(hidden.data(), embeds + (size_t)t * d, (size_t)d * sizeof(float));
        for (int l = 0; l < n_layers; l++) {
            tslm_layer_step(ctx, l, hidden.data(), t, cpu_be);
            if (t < hooks.max_capture_positions) {
                if (l == hooks.layer0_capture && hooks.layer0_out) {
                    hooks.layer0_out->insert(hooks.layer0_out->end(), hidden.data(), hidden.data() + d);
                }
                if (l == hooks.layer_last_capture && hooks.layer_last_out) {
                    hooks.layer_last_out->insert(hooks.layer_last_out->end(), hidden.data(), hidden.data() + d);
                }
            }
        }
        ctx->tslm_kv.n_past = t + 1;
        if (hooks.all_positions && t < hooks.max_capture_positions) {
            hooks.all_positions->insert(hooks.all_positions->end(), hidden.data(), hidden.data() + d);
        }
    }
    return hidden;
}

// ---------------------------------------------------------------------------
// RALM prefill — one token, fills KV cache
// ---------------------------------------------------------------------------

static std::vector<float> ralm_prefill(voxcpm2_context* ctx, const std::vector<float>& input, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.ralm_d_model;
    ctx->ralm_kv.reset();

    std::vector<float> hidden = input;
    if ((int)hidden.size() < d)
        hidden.resize(d, 0.0f);
    if ((int)hidden.size() > d)
        hidden.resize(d);

    for (int l = 0; l < (int)hp.ralm_n_layers; l++) {
        ralm_layer_step(ctx, l, hidden.data(), cpu_be);
    }
    ctx->ralm_kv.n_past = 1;

    return hidden;
}

// Multi-position RALM prefill — processes T tokens sequentially with causal attention.
// Input: [T * d] row-major (T vectors of d dimensions).
// Returns: [T * d] row-major output hidden states (pre-output-norm).
static std::vector<float> ralm_prefill_multi(voxcpm2_context* ctx, const float* input, int T, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.ralm_d_model;
    ctx->ralm_kv.reset();

    std::vector<float> all_out((size_t)T * d);
    std::vector<float> hidden(d);

    for (int t = 0; t < T; t++) {
        std::memcpy(hidden.data(), input + (size_t)t * d, (size_t)d * sizeof(float));

        for (int l = 0; l < (int)hp.ralm_n_layers; l++) {
            ralm_layer_step(ctx, l, hidden.data(), cpu_be);
        }
        ctx->ralm_kv.n_past = t + 1;

        std::memcpy(all_out.data() + (size_t)t * d, hidden.data(), (size_t)d * sizeof(float));
    }

    return all_out;
}

// ---------------------------------------------------------------------------
// FSQ: Linear -> tanh -> round(x*9)/9 -> Linear
// ---------------------------------------------------------------------------

static std::vector<float> fsq_forward(voxcpm2_context* ctx, const float* x, ggml_backend_t cpu_be) {
    const vox_weights& W = ctx->weights;
    int d_in = 2048;
    int d_mid = 512;
    int d_out = 2048;

    std::vector<float> mid(d_mid), quantized(d_mid), out(d_out);

    matmul_mv_bias(cpu_be, W.fsq_in_proj_w, W.fsq_in_proj_b, x, d_in, mid.data(), d_mid);
    for (int i = 0; i < d_mid; i++) {
        float v = std::tanh(mid[i]);
        quantized[i] = std::round(v * 9.0f) / 9.0f;
    }
    matmul_mv_bias(cpu_be, W.fsq_out_proj_w, W.fsq_out_proj_b, quantized.data(), d_mid, out.data(), d_out);

    return out;
}

// ---------------------------------------------------------------------------
// LocEnc forward — bidirectional 12-layer transformer with CLS prepend
// Input: patch flattened [patch_frames * patch_dim]
// Returns: CLS token output [d_locenc]
// ---------------------------------------------------------------------------

static std::vector<float> locenc_forward(voxcpm2_context* ctx, const float* patch, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W = ctx->weights;
    int d = (int)hp.locenc_d_model; // 1024
    int n_q = (int)hp.locenc_n_heads;
    int n_kv = (int)hp.locenc_n_kv;
    int hd = (int)hp.locenc_head_dim;
    float eps = hp.rms_norm_eps;
    float ascale = 1.0f / std::sqrt((float)hd);

    // Sequence: [CLS, patch_frames tokens]
    // PyTorch: x=[B,T,P,D=64], in_proj projects D=64→d=1024
    // Here: patch is [patch_frames * feat_dim=64], one patch of P=patch_frames frames
    int n_patch_tok = (int)hp.patch_frames; // P = 4 frames per patch
    int T = n_patch_tok + 1;                // CLS + P tokens
    int feat_dim = 64;                      // VAE latent dim per time-frame

    std::vector<float> cur((size_t)T * d, 0.0f);

    // CLS token (d-dimensional learned vector)
    if (W.locenc_cls_token) {
        std::memcpy(cur.data(), tensor_data_f32(W.locenc_cls_token), (size_t)d * sizeof(float));
    }

    // Patch tokens: project each feat_dim=64 frame to d=1024 via in_proj
    for (int f = 0; f < n_patch_tok; f++) {
        float* dst = cur.data() + (size_t)(f + 1) * d;
        const float* src = patch + (size_t)f * feat_dim;
        if (W.locenc_in_proj_w && W.locenc_in_proj_b) {
            // in_proj.weight: [feat_dim, d] stored as [d, feat_dim] in GGUF convention
            // matmul_mv_bias: W[rows=d, cols=feat_dim] x v[feat_dim] -> out[d]
            matmul_mv_bias(cpu_be, W.locenc_in_proj_w, W.locenc_in_proj_b, src, feat_dim, dst, d);
        } else {
            // fallback: copy what fits, zero-pad rest
            int copy_d = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)copy_d * sizeof(float));
        }
    }

    std::vector<float> normed((size_t)T * d), attn_out((size_t)T * d), ffn_h(d);

    // LongRoPE factors for LocEnc (same as TSLM/LocDiT)
    const float* rope_factors = W.tslm_rope_short ? tensor_data_f32(W.tslm_rope_short) : nullptr;
    float rope_theta = hp.tslm_rope_theta;

    for (int l = 0; l < (int)hp.locenc_n_layers; l++) {
        const vox_enc_layer& L = W.locenc_layers[l];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + t * d, tensor_data_f32(L.norm1_w), normed.data() + t * d, d, eps);
        }

        bidir_attn_full(normed.data(), T, d, L.attn_q_w, L.attn_k_w, L.attn_v_w, L.attn_o_w, n_q, n_kv, hd, ascale,
                        cpu_be, attn_out.data(), rope_factors, rope_theta);

        for (size_t i = 0; i < (size_t)T * d; i++)
            cur[i] += attn_out[i];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + t * d, tensor_data_f32(L.norm2_w), normed.data() + t * d, d, eps);
            swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data() + t * d, d,
                           (int)hp.locenc_ff_dim, d, ffn_h.data());
            float* ct = cur.data() + t * d;
            for (int i = 0; i < d; i++)
                ct[i] += ffn_h[i];
        }
    }

    // Return CLS token (position 0) after optional final norm
    std::vector<float> cls_out(d);
    if (W.locenc_norm_w) {
        rms_norm_cpu(cur.data(), tensor_data_f32(W.locenc_norm_w), cls_out.data(), d, eps);
    } else {
        std::memcpy(cls_out.data(), cur.data(), (size_t)d * sizeof(float));
    }

    return cls_out;
}

// ---------------------------------------------------------------------------
// LocEnc — per-call ggml_cgraph variant.
//
// Mirrors `build_locdit_graph`'s structure: bidirectional 12-layer
// transformer with LongRoPE GQA flash-attn + SwiGLU, but simpler —
// no time/dt embeddings, no mu condition, no cond projection, no
// final-P slice. Input is a single P=4-frame patch [feat_dim, P];
// the CLS-prepended sequence is T=5 long; output is just the CLS
// token at position 0 after the final norm.
//
// Gated on `VOXCPM2_USE_GRAPH=1`. The graph topology is constant
// (T=5, P=4 are model hparams), so it's cacheable with
// gallocr_reserve in the same one-shot pattern as LocDiT.
// ---------------------------------------------------------------------------

static ggml_cgraph* build_locenc_graph(voxcpm2_context* ctx, ggml_context* arena_ctx = nullptr) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W = ctx->graph_weights();
    const int d = (int)hp.locenc_d_model;
    const int n_q = (int)hp.locenc_n_heads;
    const int n_kv = (int)hp.locenc_n_kv;
    const int hd = (int)hp.locenc_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float ascale = 1.0f / std::sqrt((float)hd);
    const int feat_dim = 64;
    const int P = (int)hp.patch_frames; // 4
    const int T = P + 1;                // CLS + P frames

    ggml_context* ctx0 = arena_ctx;
    if (!ctx0) {
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), /*no_alloc=*/true};
        ctx0 = ggml_init(ip);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // ── Inputs ───────────────────────────────────────────────────
    ggml_tensor* patch_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, feat_dim, P);
    ggml_set_name(patch_in, "patch_in");
    ggml_set_input(patch_in);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ── in_proj on patch frames → [d, P] ─────────────────────────
    ggml_tensor* patch_proj = ggml_mul_mat(ctx0, W.locenc_in_proj_w, patch_in);
    if (W.locenc_in_proj_b) {
        patch_proj = ggml_add(ctx0, patch_proj, W.locenc_in_proj_b);
    }

    // ── Prepend CLS token (1D [d] → [d, 1]) ──────────────────────
    // `locenc_cls_token` is stored as a 1D [d] tensor. We view it as
    // [d, 1] before the concat. If it's absent, the legacy path would
    // start from zeros at position 0; mirror that with a zero tensor.
    ggml_tensor* cls_2d = nullptr;
    if (W.locenc_cls_token) {
        cls_2d = ggml_reshape_2d(ctx0, W.locenc_cls_token, d, 1);
    } else {
        // Build a zero CLS via an explicit input (rare path).
        cls_2d = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 1);
        ggml_set_name(cls_2d, "cls_zero");
        ggml_set_input(cls_2d);
    }
    ggml_tensor* cur = ggml_concat(ctx0, cls_2d, patch_proj, /*dim=*/1); // [d, T=5]

    const float* rope_factors_ptr = nullptr;
    (void)rope_factors_ptr;
    ggml_tensor* rope_factors = W.tslm_rope_short; // LocEnc reuses TSLM's LongRoPE factors

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.tslm_max_pos,
        /*rope_theta*/ hp.tslm_rope_theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ ascale,
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
        /*rope_type*/ GGML_ROPE_TYPE_NEOX,
        /*n_rot*/ 0,
        /*v_rms_norm*/ false,
        /*rope_freq_factors*/ rope_factors,
    };

    // ── 12 bidirectional transformer layers ──────────────────────
    for (uint32_t il = 0; il < hp.locenc_n_layers; il++) {
        const vox_enc_layer& L = W.locenc_layers[il];
        ggml_tensor* residual = cur;

        // Pre-attn RMSNorm + scale (norm1)
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, L.norm1_w);

        // Q/K/V projections, reshape, RoPE, GQA expand, permute, flash-attn
        ggml_tensor* Q = ggml_mul_mat(ctx0, L.attn_q_w, x);
        ggml_tensor* K = ggml_mul_mat(ctx0, L.attn_k_w, x);
        ggml_tensor* V = ggml_mul_mat(ctx0, L.attn_v_w, x);

        Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        Q = ggml_rope_ext(ctx0, Q, positions, kvp.rope_freq_factors, hd, GGML_ROPE_TYPE_NEOX, kvp.n_ctx_orig,
                          kvp.rope_theta, /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f, /*attn_factor*/ 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, kvp.rope_freq_factors, hd, GGML_ROPE_TYPE_NEOX, kvp.n_ctx_orig,
                          kvp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        if (n_kv_grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            ggml_tensor* V4 = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            K4 = ggml_repeat_4d(ctx0, K4, hd, n_kv_grp, n_kv, T);
            V4 = ggml_repeat_4d(ctx0, V4, hd, n_kv_grp, n_kv, T);
            K = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, n_q, T));
            V = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, n_q, T));
        }

        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // Bidirectional flash-attn (no mask). PREC_F32 NOT set — Metal
        // refuses FA ops tagged PREC_F32 (see LocDiT graph for rationale).
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, /*mask=*/nullptr, ascale, /*max_bias*/ 0.0f,
                                                /*logit_softcap*/ 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        attn = ggml_mul_mat(ctx0, L.attn_o_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // Pre-FFN RMSNorm × scale, SwiGLU, residual
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, L.norm2_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // ── Final norm + extract CLS (position 0) ────────────────────
    // cur is [d, T=5]; view position 0 as [d, 1].
    ggml_tensor* cls_view = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], /*offset=*/0);
    ggml_tensor* cls = cls_view;
    if (W.locenc_norm_w) {
        cls = ggml_rms_norm(ctx0, cls_view, eps);
        cls = ggml_mul(ctx0, cls, W.locenc_norm_w);
    }
    ggml_set_name(cls, "cls_out");
    ggml_set_output(cls);
    ggml_build_forward_expand(gf, cls);

    if (!arena_ctx) {
        ggml_free(ctx0);
    }
    return gf;
}

static ggml_cgraph* get_or_build_locenc_graph(voxcpm2_context* ctx) {
    if (ctx->locenc_gf) {
        return ctx->locenc_gf;
    }
    if (!ctx->backend) {
        return nullptr;
    }
    ctx->locenc_arena_meta.assign(ctx->compute_meta.size(), 0);
    ggml_init_params ip = {ctx->locenc_arena_meta.size(), ctx->locenc_arena_meta.data(), /*no_alloc=*/true};
    ctx->locenc_arena_ctx = ggml_init(ip);
    if (!ctx->locenc_arena_ctx) {
        ctx->locenc_arena_meta.clear();
        return nullptr;
    }
    ctx->locenc_gf = build_locenc_graph(ctx, ctx->locenc_arena_ctx);
    if (!ctx->locenc_gf) {
        ggml_free(ctx->locenc_arena_ctx);
        ctx->locenc_arena_ctx = nullptr;
        ctx->locenc_arena_meta.clear();
        return nullptr;
    }
    ctx->locenc_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->locenc_galloc || !ggml_gallocr_reserve(ctx->locenc_galloc, ctx->locenc_gf)) {
        if (ctx->locenc_galloc) {
            ggml_gallocr_free(ctx->locenc_galloc);
            ctx->locenc_galloc = nullptr;
        }
        ggml_free(ctx->locenc_arena_ctx);
        ctx->locenc_arena_ctx = nullptr;
        ctx->locenc_gf = nullptr;
        ctx->locenc_arena_meta.clear();
        return nullptr;
    }
    return ctx->locenc_gf;
}

// Run LocEnc through the graph. Same signature shape as `locenc_forward`
// (single patch in, CLS hidden state out). Falls back to the legacy CPU
// path if graph init fails.
static std::vector<float> locenc_forward_graph(voxcpm2_context* ctx, const float* patch) {
    const int d = (int)ctx->hp.locenc_d_model;
    const int feat_dim = 64;
    const int P = (int)ctx->hp.patch_frames;
    const int T = P + 1;

    std::vector<float> patch_buf(patch, patch + feat_dim * P);
    std::vector<int32_t> positions(T);
    for (int i = 0; i < T; i++)
        positions[i] = i;

    ggml_cgraph* gf = get_or_build_locenc_graph(ctx);
    if (!gf) {
        return locenc_forward(ctx, patch, ctx->backend_cpu);
    }
    if (!ggml_gallocr_alloc_graph(ctx->locenc_galloc, gf)) {
        fprintf(stderr, "voxcpm2: locenc gallocr alloc failed\n");
        return locenc_forward(ctx, patch, ctx->backend_cpu);
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "patch_in"), patch_buf.data(), 0,
                            patch_buf.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));

    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, g_cpu_n_threads);
    }
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxcpm2: locenc graph compute failed\n");
        return locenc_forward(ctx, patch, ctx->backend_cpu);
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "cls_out");
    std::vector<float> result(d);
    ggml_backend_tensor_get(out, result.data(), 0, (size_t)d * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Sinusoidal time embedding
// (bf16_round + BF16-noise helpers come from core/torch_rng.h via the
// using-declarations near the top of this file.)
// ---------------------------------------------------------------------------

static std::vector<float> sinusoidal_time_emb(float t_scalar, int dim) {
    // Matches Python SinusoidalPosEmb(dim).forward(x, scale=1000) in BF16:
    //   half_dim = dim // 2
    //   emb = log(10000) / (half_dim - 1)               ← Python float (F64), NOT bf16
    //   emb = exp(arange(half_dim, dtype=bf16) * -emb)  ← arange is bf16; mul is bf16(F32*F32)
    //   emb = scale * x.unsqueeze(1) * emb              ← scale=1000 (exact), x is bf16
    //   return cat(sin(emb), cos(emb))
    // Verified bit-identical to torch SinusoidalPosEmb(bf16) for half_dim=512.
    int half_dim = dim / 2;
    double log_base = std::log(10000.0) / (double)(half_dim - 1); // F64, not bf16-rounded
    float scale_val = bf16_round(1000.0f);                        // 1000 is exact in bf16
    float x_val = bf16_round(t_scalar);
    std::vector<float> emb(dim, 0.0f);
    for (int i = 0; i < half_dim; i++) {
        float i_bf = bf16_round((float)i); // arange(dtype=bf16) loses precision for i>256
        // freq = bf16(exp(bf16(i_bf * (-log_base))))
        float freq = bf16_round(std::exp(bf16_round((float)(-(double)i_bf * log_base))));
        // scale*x then *freq — two bf16 rounds matching Python's chained ops
        float val = bf16_round(bf16_round(scale_val * x_val) * freq);
        emb[i] = bf16_round(std::sin(val));            // first half: sin
        emb[i + half_dim] = bf16_round(std::cos(val)); // second half: cos
    }
    return emb;
}

// ---------------------------------------------------------------------------
// LocDiT forward — bidirectional 12-layer DiT, single denoising step
//
// PyTorch signature:
//   x:    (N, C=64, T=patch_size) noisy latent — here patch_size=patch_frames=4
//   mu:   (N, hidden=2048)         conditioning from TSLM+RALM
//   t:    (N,)                     diffusion timestep in [0,1]
//   cond: (N, C=64, T'=patch_size) previous patch condition (zeros for first step)
//   dt:   (N,)                     delta time (zeros for non-mean-mode)
//
// Sequence layout:
//   [mu_reshaped (2 tokens, d=1024), t_emb (1 token), cond_proj (P tokens), x_proj (P tokens)]
//   total T_seq = 2 + 1 + P + P
//
// We extract only the x-portion (last P tokens) and out-project to feat_dim=64.
//
// Arguments here (simplified to single-batch CPU):
//   x_raw:     [feat_dim * patch_frames] = [64 * 4 = 256] noisy latent
//   mu:        [tslm_d_model = 2048] conditioning
//   t_scalar:  timestep scalar in [0,1]
//   cond_raw:  [feat_dim * patch_frames] = [256] previous patch (zeros if first step)
//   dt_scalar: delta time (0.0 for non-mean-mode)
// Returns:     [feat_dim * patch_frames] = [256] predicted velocity
// ---------------------------------------------------------------------------

static std::vector<float> locdit_forward(voxcpm2_context* ctx, const float* x_raw, const float* mu, float t_scalar,
                                         const float* cond_raw, float dt_scalar, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W = ctx->weights;
    int d = (int)hp.locdit_d_model; // 1024
    int n_q = (int)hp.locdit_n_heads;
    int n_kv = (int)hp.locdit_n_kv;
    int hd = (int)hp.locdit_head_dim;
    float eps = hp.rms_norm_eps;
    float ascale = 1.0f / std::sqrt((float)hd);
    int feat_dim = 64;
    int P = (int)hp.patch_frames; // 4 frames per patch
    int mu_toks = 2;              // 2048 / 1024 = 2 mu tokens
    // Total sequence: [mu_toks, 1_time, P_cond, P_x] = 2+1+4+4 = 11 tokens
    int T = mu_toks + 1 + P + P;
    int x_offset = mu_toks + 1 + P; // position of x tokens in sequence

    // --- Build time embedding: sinusoidal -> two-layer MLP (Linear→SiLU→Linear) ---
    std::vector<float> t_sin = sinusoidal_time_emb(t_scalar, d);
    std::vector<float> dt_sin = sinusoidal_time_emb(dt_scalar, d);

    // time_mlp: Linear(d,d) -> SiLU -> Linear(d,d)
    auto two_layer_mlp = [&](ggml_tensor* w0, ggml_tensor* b0, ggml_tensor* w1, ggml_tensor* b1,
                             const std::vector<float>& inp) -> std::vector<float> {
        std::vector<float> h0(d), h1(d);
        if (w0 && b0) {
            matmul_mv_bias(cpu_be, w0, b0, inp.data(), d, h0.data(), d);
        } else {
            h0 = inp;
        }
        // SiLU activation: x * sigmoid(x)
        for (int i = 0; i < d; i++) {
            float x = h0[i];
            h0[i] = x / (1.0f + std::exp(-x));
        }
        if (w1 && b1) {
            matmul_mv_bias(cpu_be, w1, b1, h0.data(), d, h1.data(), d);
        } else {
            h1 = h0;
        }
        return h1;
    };

    std::vector<float> t_emb = two_layer_mlp(W.locdit_time_mlp_0_w, W.locdit_time_mlp_0_b, W.locdit_time_mlp_1_w,
                                             W.locdit_time_mlp_1_b, t_sin);
    std::vector<float> dt_emb =
        two_layer_mlp(W.locdit_dt_mlp_0_w, W.locdit_dt_mlp_0_b, W.locdit_dt_mlp_1_w, W.locdit_dt_mlp_1_b, dt_sin);
    // t_emb = t_emb + dt_emb
    for (int i = 0; i < d; i++)
        t_emb[i] += dt_emb[i];

    // --- Build sequence ---
    std::vector<float> cur((size_t)T * d, 0.0f);

    // Tokens 0..1: mu reshaped [2048] -> 2 x [1024]
    std::memcpy(cur.data(), mu, (size_t)d * sizeof(float));
    std::memcpy(cur.data() + d, mu + d, (size_t)d * sizeof(float));

    // Token 2: time embedding [1024]
    std::memcpy(cur.data() + 2 * d, t_emb.data(), (size_t)d * sizeof(float));

    // Tokens 3..3+P-1: cond_proj applied to each of P frames of cond_raw
    for (int p = 0; p < P; p++) {
        float* dst = cur.data() + (size_t)(mu_toks + 1 + p) * d;
        const float* src = cond_raw + (size_t)p * feat_dim;
        if (W.locdit_cond_proj_w && W.locdit_cond_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_cond_proj_w, W.locdit_cond_proj_b, src, feat_dim, dst, d);
        } else {
            int cp = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)cp * sizeof(float));
        }
    }

    // Tokens 3+P..3+2P-1: in_proj applied to each of P frames of x_raw
    for (int p = 0; p < P; p++) {
        float* dst = cur.data() + (size_t)(x_offset + p) * d;
        const float* src = x_raw + (size_t)p * feat_dim;
        if (W.locdit_in_proj_w && W.locdit_in_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_in_proj_w, W.locdit_in_proj_b, src, feat_dim, dst, d);
        } else {
            int cp = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)cp * sizeof(float));
        }
    }

    // --- Bidirectional transformer layers ---
    // LongRoPE factors for LocDiT (same as TSLM/LocEnc)
    const float* rope_factors = W.tslm_rope_short ? tensor_data_f32(W.tslm_rope_short) : nullptr;
    float rope_theta = hp.tslm_rope_theta;

    std::vector<float> normed((size_t)T * d), attn_out((size_t)T * d), ffn_h(d);

    for (int l = 0; l < (int)hp.locdit_n_layers; l++) {
        const vox_enc_layer& L = W.locdit_layers[l];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + (size_t)t * d, tensor_data_f32(L.norm1_w), normed.data() + (size_t)t * d, d, eps);
        }

        bidir_attn_full(normed.data(), T, d, L.attn_q_w, L.attn_k_w, L.attn_v_w, L.attn_o_w, n_q, n_kv, hd, ascale,
                        cpu_be, attn_out.data(), rope_factors, rope_theta);

        for (size_t i = 0; i < (size_t)T * d; i++)
            cur[i] += attn_out[i];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + (size_t)t * d, tensor_data_f32(L.norm2_w), normed.data() + (size_t)t * d, d, eps);
            swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data() + (size_t)t * d, d,
                           (int)hp.locdit_ff_dim, d, ffn_h.data());
            float* ct = cur.data() + (size_t)t * d;
            for (int i = 0; i < d; i++)
                ct[i] += ffn_h[i];
        }
    }

    // --- Extract x-portion and apply final norm + out_proj ---
    // Take only the P tokens corresponding to x (positions x_offset..x_offset+P-1)
    // Output: [feat_dim * P] = [64 * 4 = 256] predicted velocity
    int out_size = feat_dim * P;
    std::vector<float> vel(out_size);

    for (int p = 0; p < P; p++) {
        const float* h_p = cur.data() + (size_t)(x_offset + p) * d;
        float* v_p = vel.data() + (size_t)p * feat_dim;
        std::vector<float> normed_p(d);
        if (W.locdit_norm_w) {
            rms_norm_cpu(h_p, tensor_data_f32(W.locdit_norm_w), normed_p.data(), d, eps);
        } else {
            std::memcpy(normed_p.data(), h_p, (size_t)d * sizeof(float));
        }
        // out_proj: [d, feat_dim] stored as [feat_dim, d] in GGUF → matmul gives [feat_dim]
        if (W.locdit_out_proj_w && W.locdit_out_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_out_proj_w, W.locdit_out_proj_b, normed_p.data(), d, v_p, feat_dim);
        } else {
            int cp = std::min(d, feat_dim);
            std::memcpy(v_p, normed_p.data(), (size_t)cp * sizeof(float));
        }
    }

    return vel;
}

// ---------------------------------------------------------------------------
// LocDiT — per-call ggml_cgraph variant.
//
// Replaces the ~30 per-matmul tiny graphs (`matmul_mv_ggml`) per
// `locdit_forward` invocation with one cgraph for the full 12-layer
// bidirectional DiT. Same algebra as `locdit_forward`, gated on
// `VOXCPM2_USE_GRAPH=1`.
//
// Inputs (all set per call via ggml_backend_tensor_set):
//   x_in     [feat_dim=64, P=4]  F32  noisy latent
//   cond_in  [feat_dim=64, P=4]  F32  previous patch (zeros at step 0)
//   mu_in    [d=1024, mu_toks=2] F32  TSLM/RALM-projected mu
//   t_sin    [d]                 F32  bf16-rounded sinusoidal time emb
//   dt_sin   [d]                 F32  bf16-rounded sinusoidal dt emb
//
// `t_sin` / `dt_sin` are precomputed in C++ via the existing
// `sinusoidal_time_emb` so the bf16-round bug-#24 fix from commit 52622dc2
// is preserved.
//
// Output (named "vel"):
//   vel      [feat_dim=64, P=4]  F32  predicted velocity
// ---------------------------------------------------------------------------

static ggml_cgraph* build_locdit_graph(voxcpm2_context* ctx, ggml_context* arena_ctx = nullptr) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W = ctx->graph_weights();
    const int d = (int)hp.locdit_d_model;
    const int n_q = (int)hp.locdit_n_heads;
    const int n_kv = (int)hp.locdit_n_kv;
    const int hd = (int)hp.locdit_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float ascale = 1.0f / std::sqrt((float)hd);
    const int feat_dim = 64;
    const int P = (int)hp.patch_frames; // 4
    const int mu_toks = 2;
    const int T = mu_toks + 1 + P + P; // 11
    const int x_offset = mu_toks + 1 + P;

    // When arena_ctx is supplied, build into the caller's persistent arena
    // (the graph + tensor metadata outlive this call); otherwise fall back
    // to the shared compute_meta (last-write-wins, single-call lifetime).
    ggml_context* ctx0 = arena_ctx;
    if (!ctx0) {
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), /*no_alloc=*/true};
        ctx0 = ggml_init(ip);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // ── Inputs ───────────────────────────────────────────────────
    ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, feat_dim, P);
    ggml_set_name(x_in, "x_in");
    ggml_set_input(x_in);
    ggml_tensor* cond_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, feat_dim, P);
    ggml_set_name(cond_in, "cond_in");
    ggml_set_input(cond_in);
    ggml_tensor* mu_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, mu_toks);
    ggml_set_name(mu_in, "mu_in");
    ggml_set_input(mu_in);
    ggml_tensor* t_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
    ggml_set_name(t_sin, "t_sin");
    ggml_set_input(t_sin);
    ggml_tensor* dt_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
    ggml_set_name(dt_sin, "dt_sin");
    ggml_set_input(dt_sin);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ── time_mlp + delta_time_mlp ────────────────────────────────
    // Each MLP: Linear → SiLU → Linear, both with bias. Sum the two MLPs.
    auto two_layer_mlp = [&](ggml_tensor* sin_in, ggml_tensor* w0, ggml_tensor* b0, ggml_tensor* w1,
                             ggml_tensor* b1) -> ggml_tensor* {
        // sin_in is 1-D [d]; ggml_mul_mat needs a [d, T=1] 2-D view.
        ggml_tensor* in_2d = ggml_reshape_2d(ctx0, sin_in, d, 1);
        ggml_tensor* h = ggml_mul_mat(ctx0, w0, in_2d);
        h = ggml_add(ctx0, h, b0);
        h = ggml_silu(ctx0, h);
        h = ggml_mul_mat(ctx0, w1, h);
        h = ggml_add(ctx0, h, b1);
        return h; // [d, 1]
    };
    ggml_tensor* t_emb_2d = two_layer_mlp(t_sin, W.locdit_time_mlp_0_w, W.locdit_time_mlp_0_b, W.locdit_time_mlp_1_w,
                                          W.locdit_time_mlp_1_b);
    ggml_tensor* dt_emb_2d =
        two_layer_mlp(dt_sin, W.locdit_dt_mlp_0_w, W.locdit_dt_mlp_0_b, W.locdit_dt_mlp_1_w, W.locdit_dt_mlp_1_b);
    ggml_tensor* time_token = ggml_add(ctx0, t_emb_2d, dt_emb_2d); // [d, 1]

    // ── in_proj / cond_proj on x / cond ──────────────────────────
    ggml_tensor* x_proj = ggml_mul_mat(ctx0, W.locdit_in_proj_w, x_in);         // [d, P]
    x_proj = ggml_add(ctx0, x_proj, W.locdit_in_proj_b);                        // bias broadcast
    ggml_tensor* cond_proj = ggml_mul_mat(ctx0, W.locdit_cond_proj_w, cond_in); // [d, P]
    cond_proj = ggml_add(ctx0, cond_proj, W.locdit_cond_proj_b);

    // ── Concat to [d, T=11] in order [mu_toks(2) | time(1) | cond(P) | x(P)] ──
    ggml_tensor* mu_time = ggml_concat(ctx0, mu_in, time_token, /*dim=*/1); // [d, 3]
    ggml_tensor* mu_time_cond = ggml_concat(ctx0, mu_time, cond_proj, 1);   // [d, 3+P=7]
    ggml_tensor* cur = ggml_concat(ctx0, mu_time_cond, x_proj, 1);          // [d, T=11]

    // LongRoPE freq factors as a graph input (zero-copy view into the F32
    // weight tensor). NEOX RoPE expects [n_rot/2] factors; tslm_rope_short
    // matches that shape.
    ggml_tensor* rope_factors = W.tslm_rope_short; // F32 weight tensor; nullptr → no factors

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.tslm_max_pos,
        /*rope_theta*/ hp.tslm_rope_theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ ascale,
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
        /*rope_type*/ GGML_ROPE_TYPE_NEOX,
        /*n_rot*/ 0,
        /*v_rms_norm*/ false,
        /*rope_freq_factors*/ rope_factors,
    };

    // ── 12 bidirectional transformer layers ──────────────────────
    // Bidirectional == no causal mask. No KV cache (full T each call).
    // We can't reuse core_attn::kv_self_attn (it requires a KV cache);
    // inline the smaller bidir-attn path here.
    for (uint32_t il = 0; il < hp.locdit_n_layers; il++) {
        const vox_enc_layer& L = W.locdit_layers[il];
        ggml_tensor* residual = cur;

        // RMSNorm + scale (norm1)
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, L.norm1_w);

        // Q/K/V projections (no biases on LocDiT/LocEnc attn)
        ggml_tensor* Q = ggml_mul_mat(ctx0, L.attn_q_w, x);
        ggml_tensor* K = ggml_mul_mat(ctx0, L.attn_k_w, x);
        ggml_tensor* V = ggml_mul_mat(ctx0, L.attn_v_w, x);

        Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        // LongRoPE (NEOX) with tslm_rope_short factors. Same theta /
        // n_ctx_orig as TSLM (LocDiT inherits MiniCPM RoPE config via
        // model_copy(deep=True)).
        Q = ggml_rope_ext(ctx0, Q, positions, kvp.rope_freq_factors, hd, GGML_ROPE_TYPE_NEOX, kvp.n_ctx_orig,
                          kvp.rope_theta, /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f, /*attn_factor*/ 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, kvp.rope_freq_factors, hd, GGML_ROPE_TYPE_NEOX, kvp.n_ctx_orig,
                          kvp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // GQA: LocDiT has n_q == n_kv (16/16), so n_kv_grp == 1 — the
        // expansion is a no-op for this architecture. Keep the branch
        // out for clarity and future-proofing if hp ever changes.
        if (n_kv_grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            ggml_tensor* V4 = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            K4 = ggml_repeat_4d(ctx0, K4, hd, n_kv_grp, n_kv, T);
            V4 = ggml_repeat_4d(ctx0, V4, hd, n_kv_grp, n_kv, T);
            K = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, n_q, T));
            V = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, n_q, T));
        }

        // Permute to flash-attention layout: (head_dim, T, n_heads)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // Bidirectional flash-attn (no mask). PREC_F32 NOT set here —
        // Metal's `supports_op` for FLASH_ATTN_EXT refuses any op tagged
        // PREC_F32 (the chatterbox patch — gpu accumulator drift work),
        // so leaving the default lets Metal pick its native F16 simdgroup
        // path. The diff-harness gate (cfm_step0_result cos_mean ≥ 0.93)
        // tolerates the resulting drift; bit-identical CPU vs Metal isn't
        // required for voxcpm2.
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, /*mask=*/nullptr, ascale, /*max_bias*/ 0.0f,
                                                /*logit_softcap*/ 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        // Output projection (no bias on attn)
        attn = ggml_mul_mat(ctx0, L.attn_o_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // Pre-FFN norm + scale, SwiGLU, residual
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, L.norm2_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // ── Final norm + out_proj on last P positions ────────────────
    // Slice positions [x_offset, x_offset+P) out of [d, T=11]. View is
    // contiguous along d, strided along T (which is already contiguous
    // for cur), so ggml_view_2d works directly without ggml_cont.
    ggml_tensor* x_tail = ggml_view_2d(ctx0, cur, d, P, cur->nb[1], (size_t)x_offset * cur->nb[1]);
    ggml_tensor* normed = ggml_rms_norm(ctx0, x_tail, eps);
    if (W.locdit_norm_w) {
        normed = ggml_mul(ctx0, normed, W.locdit_norm_w);
    }
    ggml_tensor* vel = ggml_mul_mat(ctx0, W.locdit_out_proj_w, normed); // [feat_dim, P]
    vel = ggml_add(ctx0, vel, W.locdit_out_proj_b);
    ggml_set_name(vel, "vel");
    ggml_set_output(vel);
    ggml_build_forward_expand(gf, vel);

    // Caller-supplied arena outlives this function; only free the local one.
    if (!arena_ctx) {
        ggml_free(ctx0);
    }
    return gf;
}

// Build / fetch the cached LocDiT cgraph. Topology is invariant across
// calls (no n_past, no KV cache), so the graph + gallocr layout live for
// the lifetime of the context. Returns nullptr on failure (caller falls
// back to the per-call build path).
static ggml_cgraph* get_or_build_locdit_graph(voxcpm2_context* ctx) {
    if (ctx->locdit_gf) {
        return ctx->locdit_gf;
    }
    if (!ctx->backend) {
        return nullptr;
    }
    // Dedicated arena — `ctx->compute_meta` is shared with other graphs
    // (e.g. dynamic-path TSLM step) that would otherwise stomp on the
    // cached LocDiT tensor metadata.
    ctx->locdit_arena_meta.assign(ctx->compute_meta.size(), 0);
    ggml_init_params ip = {ctx->locdit_arena_meta.size(), ctx->locdit_arena_meta.data(), /*no_alloc=*/true};
    ctx->locdit_arena_ctx = ggml_init(ip);
    if (!ctx->locdit_arena_ctx) {
        ctx->locdit_arena_meta.clear();
        return nullptr;
    }
    ctx->locdit_gf = build_locdit_graph(ctx, ctx->locdit_arena_ctx);
    if (!ctx->locdit_gf) {
        ggml_free(ctx->locdit_arena_ctx);
        ctx->locdit_arena_ctx = nullptr;
        ctx->locdit_arena_meta.clear();
        return nullptr;
    }
    ctx->locdit_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->locdit_galloc || !ggml_gallocr_reserve(ctx->locdit_galloc, ctx->locdit_gf)) {
        if (ctx->locdit_galloc) {
            ggml_gallocr_free(ctx->locdit_galloc);
            ctx->locdit_galloc = nullptr;
        }
        ggml_free(ctx->locdit_arena_ctx);
        ctx->locdit_arena_ctx = nullptr;
        ctx->locdit_gf = nullptr;
        ctx->locdit_arena_meta.clear();
        return nullptr;
    }
    return ctx->locdit_gf;
}

// Run the LocDiT graph for one denoising step. Same signature as
// `locdit_forward` so the CFM solver can swap between paths via a single
// env check. Returns [feat_dim * P] in [T, C] row-major (same as legacy).
static std::vector<float> locdit_forward_graph(voxcpm2_context* ctx, const float* x_raw, const float* mu,
                                               float t_scalar, const float* cond_raw, float dt_scalar) {
    const vox_hparams& hp = ctx->hp;
    const int d = (int)hp.locdit_d_model;
    const int feat_dim = 64;
    const int P = (int)hp.patch_frames;
    const int mu_toks = 2;
    const int T = mu_toks + 1 + P + P;

    // ── Inputs ───────────────────────────────────────────────────
    // x_raw / cond_raw arrive as [P, feat_dim] row-major (from the CFM
    // solver's transpose). The graph's x_in tensor is column-major
    // [feat_dim, P] (ne[0]=feat_dim, ne[1]=P), so the memcpy lays out
    // x_raw[t, c] at byte offset (t * feat_dim + c) — which is exactly
    // what the column-major tensor expects (ne[1] stride = feat_dim).
    std::vector<float> x_buf(x_raw, x_raw + feat_dim * P);
    std::vector<float> cond_buf(cond_raw, cond_raw + feat_dim * P);
    std::vector<float> mu_buf(mu, mu + d * mu_toks); // 2048-vector viewed as [d, 2]
    std::vector<float> t_sin = sinusoidal_time_emb(t_scalar, d);
    std::vector<float> dt_sin = sinusoidal_time_emb(dt_scalar, d);
    std::vector<int32_t> positions(T);
    for (int i = 0; i < T; i++)
        positions[i] = i;

    // Cached graph + galloc — built once on first call, reused thereafter.
    // Falls back to the per-call build path if the cache init fails.
    ggml_cgraph* gf = get_or_build_locdit_graph(ctx);
    ggml_gallocr_t galloc = ctx->locdit_galloc;
    if (!gf) {
        gf = build_locdit_graph(ctx);
        galloc = ctx->galloc;
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "voxcpm2: locdit gallocr alloc failed\n");
        return std::vector<float>(feat_dim * P, 0.0f);
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), x_buf.data(), 0, x_buf.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_in"), cond_buf.data(), 0, cond_buf.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mu_in"), mu_buf.data(), 0, mu_buf.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_sin"), t_sin.data(), 0, t_sin.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dt_sin"), dt_sin.data(), 0, dt_sin.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));

    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, g_cpu_n_threads);
    }
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxcpm2: locdit graph compute failed\n");
        return std::vector<float>(feat_dim * P, 0.0f);
    }

    ggml_tensor* vel = ggml_graph_get_tensor(gf, "vel");
    std::vector<float> out((size_t)feat_dim * P);
    ggml_backend_tensor_get(vel, out.data(), 0, out.size() * sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// CFM Euler solve — sway schedule (t: 1->0), CFG-zero-star
//
// mu:       [tslm_d_model=2048] conditioning from TSLM+RALM
// cond_raw: [feat_dim * patch_frames = 256] previous patch latents
//           (pass all-zeros for first step)
// Returns:  [feat_dim * patch_frames = 256] denoised patch in latent space
// ---------------------------------------------------------------------------

static std::vector<float> cfm_euler_solve(voxcpm2_context* ctx, const float* mu, const float* cond_raw, int steps,
                                          float cfg, ggml_backend_t cpu_be, const float* initial_noise = nullptr) {
    const bool bench = vox_env_bool("VOXCPM2_BENCH");
    const double t_cfm0 = bench ? vox_now_ms() : 0;
    double sum_locdit = 0;

    int feat_dim = 64;
    int P = (int)ctx->hp.patch_frames; // 4
    int state_size = feat_dim * P;     // 256

    // Internal state x is [C=feat_dim, T=P] channels-first (matching Python).
    // initial_noise from reference is also [C, T] channels-first.
    std::vector<float> x(state_size, 0.0f);
    if (initial_noise) {
        std::memcpy(x.data(), initial_noise, (size_t)state_size * sizeof(float));
    }

    // VOXCPM2_USE_GRAPH=1 routes locdit through the per-call cgraph
    // (build_locdit_graph + locdit_forward_graph) instead of the
    // ~30 per-matmul tiny graphs. Same algebra; one graph build/alloc
    // per locdit call instead of one per matmul.
    const bool use_graph = vox_env_bool_default_on("VOXCPM2_USE_GRAPH");
    auto locdit_call = [&](const float* x_tc, const float* mu_in, float t_cur, const float* cond_in,
                           float dt_in) -> std::vector<float> {
        if (use_graph) {
            return locdit_forward_graph(ctx, x_tc, mu_in, t_cur, cond_in, dt_in);
        }
        return locdit_forward(ctx, x_tc, mu_in, t_cur, cond_in, dt_in, cpu_be);
    };

    // Sway schedule: t_span = linspace(1, 0, steps+1) + sway*(cos(pi/2*t) - 1 + t)
    // with sway_sampling_coef = 1.0 (default in VoxCPM2). Python computes this in
    // BF16 (model dtype); emulate per-op bf16 rounding to match torch bit-exactly.
    // Verified against torch.linspace(...,dtype=bfloat16) + sway transform.
    std::vector<float> t_span(steps + 1);
    for (int i = 0; i <= steps; i++) {
        float t = bf16_round(1.0f - (float)i / (float)steps);
        float a = bf16_round((float)M_PI / 2.0f * t);
        float cos_a = bf16_round(std::cos(a));
        float c = bf16_round(cos_a - 1.0f);
        float d = bf16_round(c + t);   // (cos-1) + t in bf16
        t_span[i] = bf16_round(t + d); // sway_coef=1.0 (no-op mul)
    }

    // CFG zero-star: first N steps skip computation (use zero velocity)
    int zero_init_steps = std::max(1, (int)(steps * 0.04f)); // = 1 for steps<=25

    float dt_scalar = 0.0f; // non-mean-mode

    for (int step = 1; step <= steps; step++) {
        float t_cur = t_span[step - 1];
        float dt_val = t_cur - t_span[step]; // positive (Python uses x = x - dt * v)

        std::vector<float> dphi_dt(state_size, 0.0f);

        if (step <= zero_init_steps) {
            // CFG zero-star: first step(s) use zero velocity
            // dphi_dt stays zero
        } else if (cfg > 1.0f) {
            // CFG with zero-star scaling:
            // Transpose x from [C, T] (internal) to [T, C] for locdit_forward
            std::vector<float> x_tc(state_size);
            for (int t = 0; t < P; t++)
                for (int c = 0; c < feat_dim; c++)
                    x_tc[t * feat_dim + c] = x[c * P + t];

            double tl = bench ? vox_now_ms() : 0;
            std::vector<float> v_cond_tc = locdit_call(x_tc.data(), mu, t_cur, cond_raw, dt_scalar);
            std::vector<float> zero_mu(ctx->hp.tslm_d_model, 0.0f);
            std::vector<float> v_uncond_tc = locdit_call(x_tc.data(), zero_mu.data(), t_cur, cond_raw, dt_scalar);
            if (bench)
                sum_locdit += vox_now_ms() - tl;

            // Transpose velocities from [T, C] back to [C, T]
            std::vector<float> v_cond(state_size), v_uncond(state_size);
            for (int t = 0; t < P; t++)
                for (int c = 0; c < feat_dim; c++) {
                    v_cond[c * P + t] = v_cond_tc[t * feat_dim + c];
                    v_uncond[c * P + t] = v_uncond_tc[t * feat_dim + c];
                }

            // CFG zero-star: st_star = dot(pos, neg) / (||neg||^2 + 1e-8)
            // pos = v_cond, neg = v_uncond (cfg branch)
            double dot = 0.0, norm_sq = 0.0;
            for (int i = 0; i < state_size; i++) {
                dot += (double)v_cond[i] * (double)v_uncond[i];
                norm_sq += (double)v_uncond[i] * (double)v_uncond[i];
            }
            float st_star = (float)(dot / (norm_sq + 1e-8));

            // dphi_dt = v_uncond * st_star + cfg * (v_cond - v_uncond * st_star)
            for (int i = 0; i < state_size; i++) {
                float neg_scaled = v_uncond[i] * st_star;
                dphi_dt[i] = neg_scaled + cfg * (v_cond[i] - neg_scaled);
            }
        } else {
            // Transpose x [C, T] → [T, C] for locdit
            std::vector<float> x_tc(state_size);
            for (int t = 0; t < P; t++)
                for (int c = 0; c < feat_dim; c++)
                    x_tc[t * feat_dim + c] = x[c * P + t];
            double tl = bench ? vox_now_ms() : 0;
            auto v_tc = locdit_call(x_tc.data(), mu, t_cur, cond_raw, dt_scalar);
            if (bench)
                sum_locdit += vox_now_ms() - tl;
            // Transpose velocity [T, C] → [C, T]
            for (int t = 0; t < P; t++)
                for (int c = 0; c < feat_dim; c++)
                    dphi_dt[c * P + t] = v_tc[t * feat_dim + c];
        }

        // Euler step: x = x - dt * dphi_dt
        for (int i = 0; i < state_size; i++)
            x[i] -= dt_val * dphi_dt[i];
    }

    if (bench) {
        double total = vox_now_ms() - t_cfm0;
        fprintf(stderr, "voxcpm2[bench]:   cfm.locdit_fwd %.1f ms total (%.1f%% of cfm)  cfm.total=%.1f ms\n",
                sum_locdit, total > 0 ? 100.0 * sum_locdit / total : 0.0, total);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Stop predictor:
//   h = stop.proj(lm_hidden + bias)   [2048 → 2048]
//   logits = stop.head(silu(h))       [2048 → 2, no bias]
//   p_stop = softmax(logits)[1]        (class 1 = stop)
// ---------------------------------------------------------------------------

static float stop_score(voxcpm2_context* ctx, const float* lm_hidden, ggml_backend_t cpu_be) {
    const vox_weights& W = ctx->weights;
    if (!W.stop_proj_w || !W.stop_proj_b)
        return 0.0f;
    int d_lm = (int)ctx->hp.tslm_d_model; // 2048

    std::vector<float> h(d_lm);
    matmul_mv_bias(cpu_be, W.stop_proj_w, W.stop_proj_b, lm_hidden, d_lm, h.data(), d_lm);
    // SiLU activation (common for projection heads in MiniCPM family)
    for (int i = 0; i < d_lm; i++) {
        float v = h[i];
        h[i] = v / (1.0f + std::exp(-v));
    }

    if (!W.stop_head_w) {
        // Fallback: treat h[0] as raw logit
        return 1.0f / (1.0f + std::exp(-h[0]));
    }

    // stop.head.weight: [d_lm, 2] — matmul gives [2] logits
    float logits[2] = {0.0f, 0.0f};
    matmul_mv(cpu_be, W.stop_head_w, h.data(), d_lm, logits, 2);

    // Softmax over 2 classes, return p(stop) = logits[1]
    float max_l = std::max(logits[0], logits[1]);
    float e0 = std::exp(logits[0] - max_l);
    float e1 = std::exp(logits[1] - max_l);
    return e1 / (e0 + e1);
}

// ===========================================================================
// VAE decoder implementation
// Architecture (sequential GGUF layer numbering):
//   layer.0  : depthwise Conv1d(64,64,k=7,groups=64)  [weight_g,weight_v,bias]
//   layer.1  : pointwise Conv1d(64,2048,k=1)           [weight_g,weight_v,bias]
//   layer.2-7: upsample blocks (rates [8,6,5,2,2,2])
//              .block.0.alpha                 — Snake1d
//              .block.1.{weight_g,weight_v,bias} — CausalTransposeConv1d
//              .block.{2,3,4}.0.alpha         — Snake1d in ResUnit
//              .block.{2,3,4}.1.{weight_g,weight_v,bias} — dilated conv (dils 1,3,9)
//              .block.{2,3,4}.2.alpha         — Snake1d in ResUnit
//              .block.{2,3,4}.3.{weight_g,weight_v,bias} — 1x1 conv
//   layer.8  : final Snake1d (.alpha)
//   layer.9  : final Conv1d(32,1,k=7)         [weight_g,weight_v,bias]
//   sr_cond.{2-7}.scale_embed / bias_embed    — [channels, 4], bucket=3 for 48kHz
//
// GGUF tensor layout: weight_v stored as [k, in_ch, out_ch] (ne[0]=k, ne[1]=in_ch, ne[2]=out_ch)
//                     weight_g stored as [out_ch] (scalar per output channel)
// Weight-norm: for each (ic, oc) pair, normalize across k, scale by g[oc].
// Output for causal_conv1d: [out_ch, in_ch, k] layout.
// ===========================================================================

// ---------------------------------------------------------------------------
// Weight-norm: reconstruct W from weight_g and weight_v stored in GGUF layout.
// GGUF layout: weight_v[ki + ic*ksize + oc*ksize*in_ch]  → [k, in_ch, out_ch]
//              weight_g[oc]                               → [out_ch]
// Normalization: along dim 0 (k) for each (ic, oc) pair.
// Output layout for causal_conv1d: [out_ch, in_ch, k]
//   → w_out[ki + ic*ksize + oc*in_ch*ksize]
// ---------------------------------------------------------------------------
// Lookup or compute the wn-scaled weight for a VAE conv layer keyed on
// the GGUF prefix (e.g. "vae.dec.layer.2.block.1"). Misses run
// wn_reconstruct + insert; hits return the cached vector by reference.
// Lifetime = voxcpm2_context. VAE decode runs sequentially per synth
// call so no locking needed.
static const std::vector<float>& vae_wn_get_cached(voxcpm2_context* ctx, const std::string& prefix,
                                                   const float* weight_g, const float* weight_v, int out_ch, int in_ch,
                                                   int ksize);

static std::vector<float> wn_reconstruct(const float* weight_g, const float* weight_v, int out_ch, int in_ch,
                                         int ksize) {
    // PyTorch weight_norm (dim=0 for Conv1d): g has shape [out_ch].
    // Norm is computed per out_ch across all (in_ch * ksize) elements.
    // weight_v GGUF layout: v[ki + ic*ksize + oc*ksize*in_ch]
    // Output layout: w[ki + ic*ksize + oc*in_ch*ksize] = [out_ch, in_ch, k]
    int total = out_ch * in_ch * ksize;
    std::vector<float> w(total);
    for (int oc = 0; oc < out_ch; oc++) {
        float g = weight_g[oc];
        // Compute L2 norm across ALL (in_ch * ksize) elements for this oc
        float norm_sq = 0.0f;
        for (int ic = 0; ic < in_ch; ic++) {
            for (int ki = 0; ki < ksize; ki++) {
                float val = weight_v[ki + (size_t)ic * ksize + (size_t)oc * ksize * in_ch];
                norm_sq += val * val;
            }
        }
        float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-12f);
        float scale = g * inv_norm;
        for (int ic = 0; ic < in_ch; ic++) {
            for (int ki = 0; ki < ksize; ki++) {
                float vval = weight_v[ki + (size_t)ic * ksize + (size_t)oc * ksize * in_ch];
                w[ki + (size_t)ic * ksize + (size_t)oc * in_ch * ksize] = vval * scale;
            }
        }
    }
    return w;
}

static const std::vector<float>& vae_wn_get_cached(voxcpm2_context* ctx, const std::string& prefix,
                                                   const float* weight_g, const float* weight_v, int out_ch, int in_ch,
                                                   int ksize) {
    auto it = ctx->vae_wn_cache.find(prefix);
    if (it != ctx->vae_wn_cache.end()) {
        return it->second;
    }
    auto inserted = ctx->vae_wn_cache.emplace(prefix, wn_reconstruct(weight_g, weight_v, out_ch, in_ch, ksize));
    return inserted.first->second;
}

// ---------------------------------------------------------------------------
// Snake1d activation: x + (1/alpha) * sin(alpha * x)^2
// alpha is a per-channel learnable parameter [C].
// x_in: [C, T]  (in-place safe if x_in == x_out — each c-row is independent)
// ---------------------------------------------------------------------------
static void snake1d(const float* alpha, const float* x_in, float* x_out, int C, int T) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    // Python ref (audio_vae_v2.py:50-56):
    //   x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    // The prior C++ used `(|a|>1e-8)?1/a:1` which silently returns 1.0 for
    // tiny alphas while Python returns ~1e9 — differs only at the limit but
    // the graph path uses the Python convention so this CPU path must too.
    for (int c = 0; c < C; c++) {
        float a = alpha[c];
        float inv_a = 1.0f / (a + 1e-9f);
        for (int t = 0; t < T; t++) {
            float v = x_in[(size_t)c * T + t];
            float s = std::sin(a * v);
            x_out[(size_t)c * T + t] = v + inv_a * s * s;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal Conv1d (CPU, arbitrary dilation, padding left only)
// weight: [out_ch, in_ch/groups, ksize]  (weight-norm already applied)
// bias:   [out_ch]  (may be nullptr)
// x_in:  [in_ch, T_in]
// x_out: [out_ch, T_out]  T_out = T_in (causal padding = (ksize-1)*dilation)
// stride: typically 1 for residual units
// groups: for depthwise conv use groups=in_ch=out_ch
// ---------------------------------------------------------------------------
static void causal_conv1d(const float* weight, const float* bias, const float* x_in, float* x_out, int out_ch,
                          int in_ch, int ksize, int T_in, int stride, int dilation, int groups) {
    int T_out = T_in / stride; // causal: same length (with causal pad)
    // Causal left-padding: pad = (ksize-1)*dilation
    int pad = (ksize - 1) * dilation;
    int in_per_grp = in_ch / groups;
    int out_per_grp = out_ch / groups;

    // Depthwise (groups == in_ch == out_ch, in_per_grp == 1) gets the
    // simple loop — there's nothing to vectorise on ic_inner. Same for
    // 1x1 conv: weight stride across ic is 1 already, so the inner loop
    // is contiguous in weight (x is still strided but transpose overhead
    // dominates the small inner work). All other cases (the dilated k=7
    // residual-unit convs at the deep upsample blocks) benefit from
    // laying weight as [k, oc, ic_inner] + transposing x to [t, ic_inner]
    // so the inner ic dot product is contiguous + NEON-auto-vectorisable.
    const bool use_transpose = (in_per_grp > 1 && ksize > 1);
    if (!use_transpose) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int oc_abs = 0; oc_abs < out_ch; oc_abs++) {
            for (int ot = 0; ot < T_out; ot++) {
                int g = oc_abs / out_per_grp;
                float b_val = bias ? bias[oc_abs] : 0.0f;
                float acc = b_val;
                int it_center = ot * stride;
                for (int k = 0; k < ksize; k++) {
                    int it = it_center - pad + k * dilation;
                    if (it < 0 || it >= T_in)
                        continue;
                    for (int ic = 0; ic < in_per_grp; ic++) {
                        int ic_abs = g * in_per_grp + ic;
                        float x_val = x_in[(size_t)ic_abs * T_in + it];
                        float w_val = weight[(size_t)oc_abs * in_per_grp * ksize + (size_t)ic * ksize + k];
                        acc += x_val * w_val;
                    }
                }
                x_out[(size_t)oc_abs * T_out + ot] = acc;
            }
        }
        return;
    }

    // Transpose weight from [oc, ic_per_grp, ksize] to [ksize, oc, ic_inner]
    // (each [oc, ic_per_grp, ksize] block stays self-contained per group).
    std::vector<float> w_kio((size_t)ksize * (size_t)out_ch * (size_t)in_per_grp);
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int k = 0; k < ksize; k++) {
        for (int oc = 0; oc < out_ch; oc++) {
            float* dst = w_kio.data() + ((size_t)k * out_ch + oc) * in_per_grp;
            for (int ic = 0; ic < in_per_grp; ic++) {
                dst[ic] = weight[(size_t)oc * in_per_grp * ksize + (size_t)ic * ksize + k];
            }
        }
    }

    // Transpose x from [in_ch, T_in] to [T_in, in_ch] (within each group's
    // ic slice, contiguous in ic).
    std::vector<float> x_tic((size_t)T_in * (size_t)in_ch);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < T_in; t++) {
        float* dst = x_tic.data() + (size_t)t * in_ch;
        for (int ic = 0; ic < in_ch; ic++) {
            dst[ic] = x_in[(size_t)ic * T_in + t];
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int oc_abs = 0; oc_abs < out_ch; oc_abs++) {
        for (int ot = 0; ot < T_out; ot++) {
            int g = oc_abs / out_per_grp;
            float acc = bias ? bias[oc_abs] : 0.0f;
            int it_center = ot * stride;
            for (int k = 0; k < ksize; k++) {
                int it = it_center - pad + k * dilation;
                if (it < 0 || it >= T_in) {
                    continue;
                }
                const float* x_row = x_tic.data() + (size_t)it * in_ch + (size_t)g * in_per_grp;
                const float* w_row = w_kio.data() + ((size_t)k * out_ch + oc_abs) * in_per_grp;
                float dot = 0.0f;
                for (int ic = 0; ic < in_per_grp; ic++) {
                    dot += x_row[ic] * w_row[ic];
                }
                acc += dot;
            }
            x_out[(size_t)oc_abs * T_out + ot] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal TransposeConv1d (upsample by stride) — matches Python's
// CausalTransposeConv1d in audio_vae_v2.py.
//
// Python (verified):
//   def __init__(self, *args, padding, output_padding, **kwargs):
//       super().__init__(*args, **kwargs)   # NOTE: padding/output_padding
//                                            # are captured by named kwargs
//                                            # and NOT forwarded — PyTorch's
//                                            # ConvTranspose1d uses 0/0.
//       self.__padding = padding
//       self.__output_padding = output_padding
//   def forward(self, x):
//       return super().forward(x)[..., : -(2*P - OP)]
//
// So `super().forward(x)` is PyTorch's no-padding/no-output_padding
// transposed conv, length L_std = (T_in - 1) * stride + K. Then Python
// slices off (2P - OP) from the end. Final length:
//   L_std - (2P - OP) = (T_in - 1) * S + K - 2*ceil(S/2) + (S%2)
// For K = 2S and P = ceil(S/2): final = T_in * S (verified for
// S in {8,6,5,2}). So **each block emits exactly T_in * stride samples,
// taking positions [0, T_in*S) of the no-padding output (= a right-trim
// of (K - S) samples)**.
//
// Effective gather: y[ot] = sum_k w[k] * x[(ot - k) / S] for ot in
// [0, T_in*S) with (ot - k) % S == 0 and (ot - k)/S in [0, T_in).
//
// weight: [in_ch, out_ch, ksize]  — note transposed layout
// x_in:  [in_ch, T_in]
// x_out: [out_ch, T_out]  T_out = T_in * stride
//
// Earlier this file used `trim = K-1` (read positions [K-1, K-1+T_in*S))
// which corresponds to a HEAD-trim instead of a TAIL-trim, shifting the
// audio by ~46 ms vs Python and producing cos=0.008 for decoded_audio in
// the diff harness. A subsequent fix-attempt tried (T_in-1)*S output —
// also wrong (slice direction confusion: Python's slice is [:-(2P-OP)]
// from a no-padding output, NOT from a padded output). The correct
// convention is TAIL-trim of (K-S) samples from the no-padding output.
// ---------------------------------------------------------------------------
static void causal_transposed_conv1d(const float* weight, const float* bias, const float* x_in, float* x_out, int in_ch,
                                     int out_ch, int ksize, int T_in, int stride) {
    int T_out = T_in * stride;
    (void)ksize; // no offset needed: take first T_in*S of the no-padding output

    // Inner ic loop in the natural layout reads x[ic*T_in+it] (stride T_in)
    // and weight[ic*out_ch*ksize+oc*ksize+k] (stride out_ch*ksize) — strided
    // on both axes, so the compiler can't auto-vectorise. Transpose once
    // per call so the inner dot product becomes contiguous + NEON-friendly:
    //   weight: [in_ch, out_ch, ksize] -> w_kio [ksize, out_ch, in_ch]
    //   x_in:   [in_ch, T_in]          -> x_tic [T_in, in_ch]
    // The transposes are O(in_ch * (out_ch * ksize + T_in)) which is small
    // compared to the inner work — block 0 (in=2048, out=1024, k=16, T=28)
    // transposes ~16 K x floats + ~32 M weight floats vs ~940 M dot-product
    // floats below. Auto-vectorisation makes the dot ~4-8× faster on M1.
    std::vector<float> w_kio((size_t)ksize * (size_t)out_ch * (size_t)in_ch);
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int k = 0; k < ksize; k++) {
        for (int oc = 0; oc < out_ch; oc++) {
            float* dst = w_kio.data() + ((size_t)k * out_ch + oc) * in_ch;
            for (int ic = 0; ic < in_ch; ic++) {
                dst[ic] = weight[((size_t)ic * out_ch + oc) * ksize + k];
            }
        }
    }

    std::vector<float> x_tic((size_t)T_in * (size_t)in_ch);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < T_in; t++) {
        float* dst = x_tic.data() + (size_t)t * in_ch;
        for (int ic = 0; ic < in_ch; ic++) {
            dst[ic] = x_in[(size_t)ic * T_in + t];
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ot = 0; ot < T_out; ot++) {
            float acc = bias ? bias[oc] : 0.0f;
            // Python convention: read no-padding transposed conv position `ot`
            // (i.e. zero offset). Valid k's: (ot - k) % S == 0 with
            // (ot - k)/S in [0, T_in). For k in [k0, K) step S where
            // k0 = ot % S.
            int k0 = ot % stride;
            for (int k = k0; k < ksize; k += stride) {
                int it = (ot - k) / stride;
                if (it < 0 || it >= T_in) {
                    continue;
                }
                const float* x_row = x_tic.data() + (size_t)it * in_ch;
                const float* w_row = w_kio.data() + ((size_t)k * out_ch + oc) * in_ch;
                float dot = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    dot += x_row[ic] * w_row[ic];
                }
                acc += dot;
            }
            x_out[(size_t)oc * T_out + ot] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: get F32 tensor data from ctx->tensors by name; returns nullptr if absent
// ---------------------------------------------------------------------------
static const float* vae_tensor_f32(const std::map<std::string, ggml_tensor*>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
        return nullptr;
    return (const float*)it->second->data;
}

static int vae_tensor_dim(const std::map<std::string, ggml_tensor*>& tensors, const std::string& name, int dim_idx) {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
        return 0;
    return (int)it->second->ne[dim_idx];
}

// ---------------------------------------------------------------------------
// CausalResidualUnit (GGUF sequential naming):
//   Tensor names follow the pattern:  prefix.0.alpha   (Snake before dilated conv)
//                                     prefix.1.{weight_g, weight_v, bias}  (dilated conv k=7)
//                                     prefix.2.alpha   (Snake before 1x1 conv)
//                                     prefix.3.{weight_g, weight_v, bias}  (1x1 conv)
//   y = x + 1x1_conv(snake(dilated_conv(snake(x))))
// ---------------------------------------------------------------------------
static void vae_residual_unit(voxcpm2_context* ctx, const std::string& prefix, const float* x_in, float* x_out, int C,
                              int T, int dilation) {
    const auto& tensors = ctx->tensors;
    // snake0 (.0.alpha)
    std::vector<float> h1((size_t)C * T);
    const float* alpha0 = vae_tensor_f32(tensors, prefix + ".0.alpha");
    if (alpha0) {
        snake1d(alpha0, x_in, h1.data(), C, T);
    } else {
        std::memcpy(h1.data(), x_in, (size_t)C * T * sizeof(float));
    }

    // dilated causal conv: .1.{weight_g, weight_v, bias}  k=7, dil=dilation
    // This is DEPTHWISE (groups=C): weight_v shape [k, 1, C] in GGUF
    int k1 = 7;
    std::vector<float> h2((size_t)C * T, 0.0f);
    const float* g1 = vae_tensor_f32(tensors, prefix + ".1.weight_g");
    const float* v1 = vae_tensor_f32(tensors, prefix + ".1.weight_v");
    const float* b1 = vae_tensor_f32(tensors, prefix + ".1.bias");
    if (g1 && v1) {
        // Depthwise: out_ch=C, in_per_grp=1, groups=C
        const auto& w1 = vae_wn_get_cached(ctx, prefix + ".1", g1, v1, C, 1, k1);
        causal_conv1d(w1.data(), b1, h1.data(), h2.data(), C, C, k1, T, 1, dilation, C);
    } else {
        std::memcpy(h2.data(), h1.data(), (size_t)C * T * sizeof(float));
    }

    // snake2 (.2.alpha)
    std::vector<float> h3((size_t)C * T);
    const float* alpha2 = vae_tensor_f32(tensors, prefix + ".2.alpha");
    if (alpha2) {
        snake1d(alpha2, h2.data(), h3.data(), C, T);
    } else {
        std::memcpy(h3.data(), h2.data(), (size_t)C * T * sizeof(float));
    }

    // 1x1 conv: .3.{weight_g, weight_v, bias}  k=1
    std::vector<float> h4((size_t)C * T, 0.0f);
    const float* g2 = vae_tensor_f32(tensors, prefix + ".3.weight_g");
    const float* v2 = vae_tensor_f32(tensors, prefix + ".3.weight_v");
    const float* b2 = vae_tensor_f32(tensors, prefix + ".3.bias");
    if (g2 && v2) {
        const auto& w2 = vae_wn_get_cached(ctx, prefix + ".3", g2, v2, C, C, 1);
        causal_conv1d(w2.data(), b2, h3.data(), h4.data(), C, C, 1, T, 1, 1, 1);
    } else {
        std::memcpy(h4.data(), h3.data(), (size_t)C * T * sizeof(float));
    }

    // residual add
    for (size_t i = 0; i < (size_t)C * T; i++)
        x_out[i] = x_in[i] + h4[i];
}

// ===========================================================================
// VOXCPM2_USE_GRAPH=1 VAE decode — full ggml cgraph (PLAN #96 follow-on)
//
// Mirrors the legacy `vae_decode` algebra but emits a single graph for the
// whole upsample stack (input convs, 6 upsample blocks with SR conditioning
// + snake + transposed conv + 3 residual units, final snake/conv/tanh).
//
// The WN-scaled conv weights, SR-cond per-bucket [C] slices, and snake1d
// `1/(α+1e-9)` reciprocals live in a dedicated arena + backend buffer
// (`vae_wn_ggml_ctx` / `vae_wn_ggml_buf`) built lazily on first invocation.
// Graph topology depends on `T_lat = n_patches * P` which varies per call,
// so we don't cache the graph itself — only the weight bridge.
// ===========================================================================

static std::vector<float> vae_decode(voxcpm2_context* ctx, const std::vector<std::vector<float>>& patches,
                                     ggml_backend_t cpu_be); // fwd-decl for fallback

// ---------------------------------------------------------------------------
// Snake1d as a ggml subgraph.
//
// Python ref (audio_vae_v2.py:50-56):
//   x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
//
// `inv_alpha` is the precomputed `1 / (α + 1e-9)` reciprocal (pre-baked at
// init time to avoid a per-call divide and to match Python's epsilon).
// `x` is [T, C] (ggml ne convention; T innermost). `alpha`/`inv_alpha`
// are [C] which we reshape to [1, C] to broadcast over T.
// ---------------------------------------------------------------------------

static ggml_tensor* snake1d_ggml(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* inv_alpha) {
    // alpha may be stored as ggml ne=[C,1,1,1] (1D) or ne=[1,C,1,1] (PyTorch
    // shape [1, C] preserved). Use ggml_nelements to be robust.
    const int C = (int)ggml_nelements(alpha);
    ggml_tensor* a2 = ggml_reshape_2d(ctx0, alpha, 1, C);
    ggml_tensor* inv2 = ggml_reshape_2d(ctx0, inv_alpha, 1, C);
    ggml_tensor* ax = ggml_mul(ctx0, x, a2);
    ggml_tensor* s = ggml_sin(ctx0, ax);
    ggml_tensor* s2 = ggml_mul(ctx0, s, s);
    ggml_tensor* term = ggml_mul(ctx0, s2, inv2);
    return ggml_add(ctx0, x, term);
}

// ---------------------------------------------------------------------------
// Causal Conv1d (left-pad K-1) as a ggml subgraph.
//
// `x` is [T_in, C_in]; `weight` is the WN-reconstructed kernel with ne
// `[K, C_in, C_out]` (forward) or `[K, 1, C]` (depthwise). `bias` is
// `[C_out]` and may be nullptr.
//
// Causality via symmetric pad + left-crop: ggml_conv_1d with `pad = (K-1)*d`
// produces `T_in + pad - K + 1 = T_in + (K-1)*(d-1)` for stride=1; we crop
// to `T_in` by taking the first T_in columns. The right-side outputs that
// we drop only depend on right-side zero-padding contributions, so the
// retained slice is causal.
// ---------------------------------------------------------------------------

static ggml_tensor* causal_conv1d_ggml(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias,
                                       int dilation, bool depthwise) {
    const int K = (int)weight->ne[0];
    const int pad = (K - 1) * dilation;
    ggml_tensor* y;
    if (depthwise) {
        y = ggml_conv_1d_dw(ctx0, weight, x, /*s*/ 1, pad, dilation);
    } else {
        y = ggml_conv_1d(ctx0, weight, x, /*s*/ 1, pad, dilation);
    }
    const int T_in = (int)x->ne[0];
    if ((int)y->ne[0] > T_in) {
        y = ggml_view_2d(ctx0, y, T_in, (int)y->ne[1], y->nb[1], 0);
        y = ggml_cont(ctx0, y);
    }
    if (bias) {
        ggml_tensor* b2d = ggml_reshape_2d(ctx0, bias, 1, (int)bias->ne[0]);
        y = ggml_add(ctx0, y, b2d);
    }
    return y;
}

// ---------------------------------------------------------------------------
// Causal Transposed Conv1d as a ggml subgraph (matches Python's
// CausalTransposeConv1d convention).
//
// Python's CausalTransposeConv1d (audio_vae_v2.py) does NOT forward
// padding/output_padding to nn.ConvTranspose1d (they're captured by named
// kwargs and never re-passed) — so PyTorch's super().forward(x) returns
// the no-padding transposed conv of length L_std = (T_in - 1)*S + K.
// The slice `[:-(2P - OP)]` then trims the TAIL by (2P - OP) samples,
// leaving the first T_in * S samples (verified empirically for
// S in {8,6,5,2,2,2}: tail trim = K - S in all cases).
//
// So: take the first T_in*S samples of `ggml_conv_transpose_1d(s, p=0, d=1)`
// (length L_std = (T_in-1)*S + K). Right-trim of (K - S) from the end.
//
// `weight` ne: [K, out_ch, in_ch] (matches wn_reconstruct layout).
// `x` ne: [T_in, in_ch]. Output ne: [T_in*S, out_ch].
// ---------------------------------------------------------------------------

static ggml_tensor* causal_transposed_conv1d_ggml(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* weight,
                                                  ggml_tensor* bias, int stride) {
    const int T_in = (int)x->ne[0];
    const int T_out = T_in * stride;
    ggml_tensor* y = ggml_conv_transpose_1d(ctx0, weight, x, stride, /*p*/ 0, /*d*/ 1);
    // y ne[0] = L_std = (T_in-1)*S + K. Take first T_in*S samples.
    const int out_ch = (int)y->ne[1];
    y = ggml_view_2d(ctx0, y, T_out, out_ch, y->nb[1], /*offset*/ 0);
    y = ggml_cont(ctx0, y);
    if (bias) {
        ggml_tensor* b2d = ggml_reshape_2d(ctx0, bias, 1, (int)bias->ne[0]);
        y = ggml_add(ctx0, y, b2d);
    }
    return y;
}

// ---------------------------------------------------------------------------
// Build the dedicated `vae_wn_ggml_*` arena + backend buffer holding all
// WN-scaled VAE decode conv weights, SR-cond per-bucket [C] slices, and
// snake1d `1/(α+1e-9)` reciprocals.
//
// Walks every VAE decoder conv layer once. Cached `vae_wn_get_cached`
// already populates `ctx->vae_wn_cache` from prior legacy decodes, but we
// don't rely on that: the init runs `wn_reconstruct` directly so the buffer
// is correct even if the legacy path hasn't run yet.
//
// Total size: a few hundred MB on the 2048-channel block 0 transposed conv
// (16 * 1024 * 2048 = 32M params × 4B = 128 MB), plus smaller layers and
// negligible alpha/SR tensors. Well within a single backend buffer.
// ---------------------------------------------------------------------------

static int vae_decoder_block_out_ch(int b) {
    static const int t[] = {1024, 512, 256, 128, 64, 32};
    return t[b];
}

static bool vae_wn_init_ggml(voxcpm2_context* ctx) {
    if (ctx->vae_wn_ggml_buf) {
        return true; // already built
    }
    if (!ctx->backend) {
        return false;
    }

    const auto& T = ctx->tensors;
    // Bail if VAE weights aren't loaded.
    if (T.find("vae.dec.layer.0.weight_g") == T.end()) {
        return false;
    }

    // Enumerate everything we need: list of (key, weight_g_name, weight_v_name,
    // out_ch, in_ch, ksize, is_conv) tuples for WN-reconstructed weights, and
    // separately the list of snake1d alpha names + SR-cond names.
    struct WnEntry {
        std::string key;    // map key, e.g. "vae.dec.layer.2.block.1"
        std::string g_name; // GGUF tensor name for weight_g
        std::string v_name; // GGUF tensor name for weight_v
        int out_ch;
        int in_ch;
        int ksize;
    };
    std::vector<WnEntry> wn_entries;
    std::vector<std::string> alpha_names; // for snake1d inv_alpha precompute
    std::vector<std::string> sr_names;    // "vae.dec.sr_cond.<b>"

    // layer.0: depthwise k=7, groups=feat_dim=64
    wn_entries.push_back({"vae.dec.layer.0", "vae.dec.layer.0.weight_g", "vae.dec.layer.0.weight_v",
                          /*out_ch*/ 64, /*in_ch*/ 1, /*ksize*/ 7});
    // layer.1: 1x1 dense, 64 -> 2048
    {
        auto it_g = T.find("vae.dec.layer.1.weight_g");
        int out_ch1 = it_g != T.end() ? (int)ggml_nelements(it_g->second) : 2048;
        wn_entries.push_back({"vae.dec.layer.1", "vae.dec.layer.1.weight_g", "vae.dec.layer.1.weight_v",
                              /*out_ch*/ out_ch1, /*in_ch*/ 64, /*ksize*/ 1});
    }

    // Upsample blocks 2..7
    static const int up_rates[] = {8, 6, 5, 2, 2, 2};
    int Cc_in = 2048; // channel count entering each block (=output of layer.1 / previous block.1)
    for (int b = 0; b < 6; b++) {
        int layer_idx = b + 2;
        std::string lp = "vae.dec.layer." + std::to_string(layer_idx);
        int out_ch_b = vae_decoder_block_out_ch(b);

        // SR cond entry (uses Cc_in channels)
        sr_names.push_back("vae.dec.sr_cond." + std::to_string(layer_idx));

        // .block.0.alpha (snake before upsample, channels = Cc_in)
        alpha_names.push_back(lp + ".block.0.alpha");

        // .block.1 transposed conv: wn_reconstruct out_ch=Cc_in, in_ch=out_ch_b, K=ne[0]
        int ksize_up = vae_tensor_dim(T, lp + ".block.1.weight_v", 0);
        if (ksize_up <= 0)
            ksize_up = 2 * up_rates[b];
        wn_entries.push_back({lp + ".block.1", lp + ".block.1.weight_g", lp + ".block.1.weight_v",
                              /*out_ch (wn arg)*/ Cc_in, /*in_ch (wn arg)*/ out_ch_b, /*ksize*/ ksize_up});

        // 3x residual units .block.{2,3,4} operating on out_ch_b channels
        for (int r = 0; r < 3; r++) {
            std::string rp = lp + ".block." + std::to_string(r + 2);
            // .0.alpha, .2.alpha (snake1d before each conv in the residual)
            alpha_names.push_back(rp + ".0.alpha");
            alpha_names.push_back(rp + ".2.alpha");
            // .1: depthwise k=7, channels=out_ch_b
            wn_entries.push_back({rp + ".1", rp + ".1.weight_g", rp + ".1.weight_v",
                                  /*out_ch*/ out_ch_b, /*in_ch*/ 1, /*ksize*/ 7});
            // .3: 1x1 dense, out_ch_b -> out_ch_b
            wn_entries.push_back({rp + ".3", rp + ".3.weight_g", rp + ".3.weight_v",
                                  /*out_ch*/ out_ch_b, /*in_ch*/ out_ch_b, /*ksize*/ 1});
        }

        Cc_in = out_ch_b;
    }

    // layer.8.alpha (final snake before layer.9), channels = Cc_in (last block out = 32)
    alpha_names.push_back("vae.dec.layer.8.alpha");
    // layer.9: k=7 dense, last_ch -> 1
    wn_entries.push_back({"vae.dec.layer.9", "vae.dec.layer.9.weight_g", "vae.dec.layer.9.weight_v",
                          /*out_ch*/ 1, /*in_ch*/ Cc_in, /*ksize*/ 7});

    // Sum tensor overhead — generous, will be exact once we count
    const size_t n_tensors_estimate = wn_entries.size() + alpha_names.size() + 2 * sr_names.size() + 16;
    const size_t meta_size = ggml_tensor_overhead() * n_tensors_estimate;

    ggml_init_params ip = {
        /*.mem_size   =*/meta_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx->vae_wn_ggml_ctx = ggml_init(ip);
    if (!ctx->vae_wn_ggml_ctx) {
        return false;
    }

    // Create the ggml_tensor metadata first (no_alloc=true), then alloc backend buffer.
    auto& M = ctx->vae_wn_ggml_tensors;

    for (const auto& e : wn_entries) {
        // Tensor shape: depthwise → ne=[K, 1, C]; otherwise ne=[K, in_ch, out_ch].
        // For the transposed-conv entries we passed out_ch (= legacy "Cc" arg) as
        // wn_reconstruct's out_ch — but wn_reconstruct emits buffer with layout
        // `[k, ic, oc]` flat = ne=[K, ic, oc]. So for ggml's transposed-conv
        // expectation `ne=[K, out_ch_layer, in_ch_layer]`, we want ne[1]=ic and
        // ne[2]=oc. ic = e.in_ch (= legacy out_ch_b = layer output channels) and
        // oc = e.out_ch (= legacy Cc = layer input channels). ✓ same memory.
        ggml_tensor* t;
        if (e.in_ch == 1 && e.ksize > 1) {
            // depthwise: ne = [K, 1, C]
            t = ggml_new_tensor_3d(ctx->vae_wn_ggml_ctx, GGML_TYPE_F32, e.ksize, 1, e.out_ch);
        } else if (e.ksize == 1) {
            // 1x1 conv: ne = [1, in_ch, out_ch]
            t = ggml_new_tensor_3d(ctx->vae_wn_ggml_ctx, GGML_TYPE_F32, 1, e.in_ch, e.out_ch);
        } else {
            // Generic forward conv (k>1, dense) OR transposed conv: ne = [K, in_ch, out_ch]
            t = ggml_new_tensor_3d(ctx->vae_wn_ggml_ctx, GGML_TYPE_F32, e.ksize, e.in_ch, e.out_ch);
        }
        if (!t) {
            return false;
        }
        ggml_set_name(t, e.key.c_str());
        M[e.key] = t;
    }

    for (const auto& name : alpha_names) {
        auto it = T.find(name);
        if (it == T.end() || !it->second)
            continue;
        int C = (int)ggml_nelements(it->second);
        ggml_tensor* t = ggml_new_tensor_1d(ctx->vae_wn_ggml_ctx, GGML_TYPE_F32, C);
        if (!t)
            return false;
        std::string key = name + ".inv";
        ggml_set_name(t, key.c_str());
        M[key] = t;
    }

    for (const auto& sr_pfx : sr_names) {
        // Each SR cond produces two [C] tensors: .sr_scale and .sr_bias.
        // GGUF scale_embed PyTorch shape is (C, 4) with bucket innermost in
        // PyTorch's C-order. The GGUF loader stores this with ggml
        // ne=[C, 4] — i.e. C innermost in memory (PyTorch's "outer" dim
        // becomes ggml's "inner" because ggml ne is reversed from PyTorch
        // shape report by gguf-py). So C = ne[0], NOT ne[1].
        // The legacy `se[c*4 + bucket]` access pattern below ALSO assumes
        // the memory layout has bucket innermost (i.e. (C, 4) row-major),
        // but since ne is reversed-but-the-data-is-the-same, both views
        // see the same flat bytes. The legacy loop uses Cc (channel count
        // from upstream) directly, sidestepping the ne-ordering ambiguity.
        auto it_s = T.find(sr_pfx + ".scale_embed");
        auto it_b = T.find(sr_pfx + ".bias_embed");
        if (it_s == T.end() || !it_s->second)
            continue;
        // Take the larger of the two ne dims — robust to either ne ordering.
        // For (C, 4) ne=[4, C] or [C, 4], C is always the non-4 dim.
        int C = (int)std::max(it_s->second->ne[0], it_s->second->ne[1]);
        ggml_tensor* ts = ggml_new_tensor_1d(ctx->vae_wn_ggml_ctx, GGML_TYPE_F32, C);
        ggml_set_name(ts, (sr_pfx + ".sr_scale").c_str());
        M[sr_pfx + ".sr_scale"] = ts;
        if (it_b != T.end() && it_b->second) {
            ggml_tensor* tb = ggml_new_tensor_1d(ctx->vae_wn_ggml_ctx, GGML_TYPE_F32, C);
            ggml_set_name(tb, (sr_pfx + ".sr_bias").c_str());
            M[sr_pfx + ".sr_bias"] = tb;
        }
    }

    // Allocate one backend buffer for all metadata tensors. On Apple Silicon
    // Metal's "shared" buffer type means the data is CPU-readable via
    // ggml_backend_tensor_get, which we don't need here (we only read inside
    // the graph) but is convenient for debugging.
    ctx->vae_wn_ggml_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx->vae_wn_ggml_ctx,
                                                                    ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->vae_wn_ggml_buf) {
        ggml_free(ctx->vae_wn_ggml_ctx);
        ctx->vae_wn_ggml_ctx = nullptr;
        M.clear();
        return false;
    }

    // Now populate. WN convs: reconstruct from g/v, write into the tensor.
    for (const auto& e : wn_entries) {
        const float* g = vae_tensor_f32(T, e.g_name);
        const float* v = vae_tensor_f32(T, e.v_name);
        if (!g || !v) {
            continue; // optional layer; leaves the tensor zero-initialised
        }
        std::vector<float> w = wn_reconstruct(g, v, e.out_ch, e.in_ch, e.ksize);
        ggml_backend_tensor_set(M[e.key], w.data(), 0, w.size() * sizeof(float));
    }

    // Snake1d inv_alpha = 1 / (α + 1e-9) per channel (matches Python's
    // (alpha + 1e-9).reciprocal() semantics — including for tiny α the
    // legacy CPU path silently rewrote as 1.0).
    for (const auto& name : alpha_names) {
        auto it = T.find(name);
        if (it == T.end() || !it->second)
            continue;
        int C = (int)ggml_nelements(it->second);
        const float* a = (const float*)it->second->data;
        std::vector<float> inv(C);
        for (int i = 0; i < C; i++) {
            inv[i] = 1.0f / (a[i] + 1e-9f);
        }
        ggml_backend_tensor_set(M[name + ".inv"], inv.data(), 0, inv.size() * sizeof(float));
    }

    // SR conditioning: scale_embed/bias_embed PyTorch shape (C, 4) — bucket
    // is the innermost axis in memory (PyTorch C-order). The legacy
    // `se[c*4 + bucket]` flat read confirms this layout regardless of how
    // gguf-py / ggml ne report the shape. C is the non-4 dim.
    const int sr_bucket = 3;
    for (const auto& sr_pfx : sr_names) {
        auto it_s = T.find(sr_pfx + ".scale_embed");
        if (it_s == T.end() || !it_s->second)
            continue;
        int C = (int)std::max(it_s->second->ne[0], it_s->second->ne[1]);
        const float* se = (const float*)it_s->second->data;
        std::vector<float> sc(C);
        for (int c = 0; c < C; c++)
            sc[c] = se[(size_t)c * 4 + sr_bucket];
        ggml_backend_tensor_set(M[sr_pfx + ".sr_scale"], sc.data(), 0, sc.size() * sizeof(float));
        if (vox_env_bool("VOXCPM2_VAE_TRACE")) {
            fprintf(stderr, "voxcpm2 VAE-trace [init] %-30s ne=[%lld,%lld] C=%d sc[0]=%.6f\n", sr_pfx.c_str(),
                    (long long)it_s->second->ne[0], (long long)it_s->second->ne[1], C, sc[0]);
        }

        auto it_b = T.find(sr_pfx + ".bias_embed");
        if (it_b != T.end() && it_b->second) {
            const float* be = (const float*)it_b->second->data;
            std::vector<float> bv(C);
            for (int c = 0; c < C; c++)
                bv[c] = be[(size_t)c * 4 + sr_bucket];
            ggml_backend_tensor_set(M[sr_pfx + ".sr_bias"], bv.data(), 0, bv.size() * sizeof(float));
        }
    }

    if (ctx->verbosity >= 1) {
        size_t total_bytes = 0;
        for (const auto& kv : M)
            total_bytes += ggml_nbytes(kv.second);
        fprintf(stderr, "voxcpm2: vae_wn ggml buffer ready (%zu tensors, %.1f MB)\n", M.size(),
                total_bytes / (1024.0 * 1024.0));
    }
    return true;
}

// ---------------------------------------------------------------------------
// VOXCPM2_USE_GRAPH=1 entry point — full VAE decode as a single cgraph.
//
// Algorithm mirrors `vae_decode` exactly; the only behaviour difference is
// numerical (ggml conv_1d / conv_transpose_1d are typically run via Metal
// on Apple Silicon, and forward/transposed conv accumulators on Metal may
// differ from CPU's auto-vectorised dot-products by ulps).
// ---------------------------------------------------------------------------

static std::vector<float> vae_decode_graph(voxcpm2_context* ctx, const std::vector<std::vector<float>>& patches) {
    int n_patches = (int)patches.size();
    if (n_patches == 0)
        return {};

    const auto& Tens = ctx->tensors;
    bool have_vae = (vae_tensor_f32(Tens, "vae.dec.layer.0.weight_g") != nullptr);
    if (!have_vae) {
        // Same graceful-silence fallback as the legacy path.
        const int P = (int)ctx->hp.patch_frames;
        return std::vector<float>((size_t)n_patches * (size_t)P * 1920, 0.0f);
    }

    if (!vae_wn_init_ggml(ctx)) {
        if (ctx->verbosity >= 1)
            fprintf(stderr, "voxcpm2: vae_wn_init_ggml failed; falling back to CPU vae_decode\n");
        return vae_decode(ctx, patches, ctx->backend_cpu);
    }

    const int feat_dim = 64;
    const int P = (int)ctx->hp.patch_frames; // 4
    const int T_lat = n_patches * P;

    // Pack patches into a flat [T_lat, feat_dim] host buffer.
    // ne convention: ne[0]=T_lat (innermost), ne[1]=feat_dim → element (t,c)
    // is at index `c * T_lat + t`.
    std::vector<float> latents_host((size_t)feat_dim * T_lat, 0.0f);
    for (int n = 0; n < n_patches; n++) {
        const auto& patch = patches[n];
        for (int p = 0; p < P; p++) {
            int t = n * P + p;
            size_t patch_off = (size_t)p * feat_dim;
            if (patch_off >= patch.size())
                break;
            const float* src = patch.data() + patch_off;
            int avail = (int)std::min((size_t)feat_dim, patch.size() - patch_off);
            for (int c = 0; c < avail; c++) {
                latents_host[(size_t)c * T_lat + t] = src[c];
            }
        }
    }

    // Build the graph (per-call — T_lat varies with n_patches).
    ggml_init_params ip = {
        /*.mem_size   =*/ctx->compute_meta.size(),
        /*.mem_buffer =*/ctx->compute_meta.data(),
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) {
        return vae_decode(ctx, patches, ctx->backend_cpu);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_lat, feat_dim);
    ggml_set_name(in, "latents");
    ggml_set_input(in);
    ggml_tensor* cur = in;

    auto& M = ctx->vae_wn_ggml_tensors;
    auto Wget = [&](const std::string& key) -> ggml_tensor* {
        auto it = M.find(key);
        return it == M.end() ? nullptr : it->second;
    };
    auto Bias = [&](const std::string& prefix) -> ggml_tensor* {
        auto it = Tens.find(prefix + ".bias");
        return (it == Tens.end()) ? nullptr : it->second;
    };
    auto Alpha = [&](const std::string& prefix) -> ggml_tensor* {
        auto it = Tens.find(prefix + ".alpha");
        return (it == Tens.end()) ? nullptr : it->second;
    };
    auto InvAlpha = [&](const std::string& prefix) -> ggml_tensor* { return Wget(prefix + ".alpha.inv"); };

    // Layer 0: depthwise k=7, channels=64
    cur = causal_conv1d_ggml(ctx0, cur, Wget("vae.dec.layer.0"), Bias("vae.dec.layer.0"),
                             /*dilation*/ 1, /*depthwise*/ true);

    const bool trace = vox_env_bool("VOXCPM2_VAE_TRACE");
    if (trace) {
        ggml_set_name(cur, "g_after_layer0");
        ggml_set_output(cur);
    }

    // Layer 1: 1x1 dense, 64 → 2048
    cur = causal_conv1d_ggml(ctx0, cur, Wget("vae.dec.layer.1"), Bias("vae.dec.layer.1"),
                             /*dilation*/ 1, /*depthwise*/ false);
    if (trace) {
        ggml_set_name(cur, "g_after_layer1");
        ggml_set_output(cur);
    }

    static const int up_rates[] = {8, 6, 5, 2, 2, 2};
    static const int dilations[] = {1, 3, 9};

    for (int b = 0; b < 6; b++) {
        const int layer_idx = b + 2;
        const std::string lp = "vae.dec.layer." + std::to_string(layer_idx);
        const std::string sr_pfx = "vae.dec.sr_cond." + std::to_string(layer_idx);

        // SR conditioning: x = x * sr_scale + sr_bias (per-channel broadcast over T)
        ggml_tensor* sr_scale = Wget(sr_pfx + ".sr_scale");
        if (sr_scale) {
            const int C = (int)sr_scale->ne[0];
            ggml_tensor* s2 = ggml_reshape_2d(ctx0, sr_scale, 1, C);
            cur = ggml_mul(ctx0, cur, s2);
            ggml_tensor* sr_bias = Wget(sr_pfx + ".sr_bias");
            if (sr_bias) {
                ggml_tensor* b2 = ggml_reshape_2d(ctx0, sr_bias, 1, C);
                cur = ggml_add(ctx0, cur, b2);
            }
        }

        if (trace && b == 0) {
            ggml_set_name(cur, "g_block_0_sr");
            ggml_set_output(cur);
        }

        // Snake before upsample
        if (Alpha(lp + ".block.0") && InvAlpha(lp + ".block.0")) {
            cur = snake1d_ggml(ctx0, cur, Alpha(lp + ".block.0"), InvAlpha(lp + ".block.0"));
        }

        if (trace && b == 0) {
            ggml_set_name(cur, "g_block_0_snake");
            ggml_set_output(cur);
        }

        // Transposed conv (upsample)
        cur = causal_transposed_conv1d_ggml(ctx0, cur, Wget(lp + ".block.1"), Bias(lp + ".block.1"), up_rates[b]);
        if (trace) {
            std::string nm = "g_block_" + std::to_string(b) + "_upsample";
            ggml_set_name(cur, nm.c_str());
            ggml_set_output(cur);
        }

        // 3x causal residual units (.block.{2,3,4} with dilations 1, 3, 9)
        for (int r = 0; r < 3; r++) {
            const std::string rp = lp + ".block." + std::to_string(r + 2);
            ggml_tensor* residual = cur;
            // snake0
            if (Alpha(rp + ".0") && InvAlpha(rp + ".0")) {
                cur = snake1d_ggml(ctx0, cur, Alpha(rp + ".0"), InvAlpha(rp + ".0"));
            }
            // dilated depthwise conv k=7
            cur = causal_conv1d_ggml(ctx0, cur, Wget(rp + ".1"), Bias(rp + ".1"), dilations[r], /*depthwise*/ true);
            // snake2
            if (Alpha(rp + ".2") && InvAlpha(rp + ".2")) {
                cur = snake1d_ggml(ctx0, cur, Alpha(rp + ".2"), InvAlpha(rp + ".2"));
            }
            // 1x1 dense
            cur = causal_conv1d_ggml(ctx0, cur, Wget(rp + ".3"), Bias(rp + ".3"), /*dilation*/ 1, /*depthwise*/ false);
            cur = ggml_add(ctx0, cur, residual);
        }
        if (trace) {
            std::string nm = "g_block_" + std::to_string(b) + "_residuals";
            ggml_set_name(cur, nm.c_str());
            ggml_set_output(cur);
        }
    }

    // Final snake (layer.8) + final conv (layer.9, K=7 → 1 channel) + tanh
    if (Alpha("vae.dec.layer.8") && InvAlpha("vae.dec.layer.8")) {
        cur = snake1d_ggml(ctx0, cur, Alpha("vae.dec.layer.8"), InvAlpha("vae.dec.layer.8"));
    }
    cur = causal_conv1d_ggml(ctx0, cur, Wget("vae.dec.layer.9"), Bias("vae.dec.layer.9"),
                             /*dilation*/ 1, /*depthwise*/ false);
    cur = ggml_tanh(ctx0, cur);
    ggml_set_name(cur, "pcm");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "voxcpm2: vae_decode_graph gallocr alloc failed; falling back to CPU\n");
        ggml_free(ctx0);
        return vae_decode(ctx, patches, ctx->backend_cpu);
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "latents"), latents_host.data(), 0,
                            latents_host.size() * sizeof(float));

    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, g_cpu_n_threads);
    }
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxcpm2: vae_decode_graph compute failed; falling back to CPU\n");
        ggml_free(ctx0);
        return vae_decode(ctx, patches, ctx->backend_cpu);
    }

    ggml_tensor* pcm_t = ggml_graph_get_tensor(gf, "pcm");
    // pcm_t ne = [T_out=T_lat*1920, out_ch=1]
    const size_t n_out = (size_t)pcm_t->ne[0] * (size_t)pcm_t->ne[1];
    std::vector<float> pcm(n_out);
    ggml_backend_tensor_get(pcm_t, pcm.data(), 0, pcm.size() * sizeof(float));

    if (vox_env_bool("VOXCPM2_VAE_TRACE")) {
        auto dump_tensor = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t)
                return;
            const size_t n = (size_t)ggml_nelements(t);
            std::vector<float> buf(n);
            ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
            float mx = 0.0f, sq = 0.0f;
            for (size_t i = 0; i < n; i++) {
                float a = std::abs(buf[i]);
                if (a > mx)
                    mx = a;
                sq += buf[i] * buf[i];
            }
            float rms = std::sqrt(sq / (float)n);
            fprintf(stderr, "voxcpm2 VAE-trace [graph] %-24s ne=[%lld,%lld] n=%zu max=%.4f rms=%.4f\n", name,
                    (long long)t->ne[0], (long long)t->ne[1], n, mx, rms);
            std::string path = std::string("/tmp/voxcpm2_") + name + ".bin";
            FILE* f = std::fopen(path.c_str(), "wb");
            if (f) {
                std::fwrite(buf.data(), sizeof(float), n, f);
                std::fclose(f);
            }
        };
        dump_tensor("g_after_layer0");
        dump_tensor("g_after_layer1");
        dump_tensor("g_block_0_sr");
        dump_tensor("g_block_0_snake");
        for (int b = 0; b < 6; b++) {
            std::string up = "g_block_" + std::to_string(b) + "_upsample";
            std::string rs = "g_block_" + std::to_string(b) + "_residuals";
            dump_tensor(up.c_str());
            dump_tensor(rs.c_str());
        }
    }

    ggml_free(ctx0);
    return pcm;
}

// ---------------------------------------------------------------------------
// VAE decode: concatenated patches -> 48kHz PCM
//
// Each patch is [feat_dim * patch_frames = 64 * 4 = 256] float values.
// We concatenate them along the time axis to form latents [64, T_latent].
// The VAE decoder upsamples by 8*6*5*2*2*2 = 1920x (from 25 Hz to 48000 Hz).
//
// Tensor naming follows the GGUF sequential layer scheme:
//   layer.0  : depthwise in-conv (k=7, groups=64)
//   layer.1  : pointwise in-conv (k=1, 64->2048)
//   layer.2-7: upsample blocks (rates [8,6,5,2,2,2])
//   layer.8  : final Snake1d
//   layer.9  : final out-conv (k=7, last_ch->1)
//   sr_cond.{2-7}.scale_embed / bias_embed : [channels, 4], bucket=3 for 48kHz
//
// When VAE weights are absent, returns silence of the correct duration.
// ---------------------------------------------------------------------------

static std::vector<float> vae_decode(voxcpm2_context* ctx, const std::vector<std::vector<float>>& patches,
                                     ggml_backend_t /*cpu_be*/) {
    int n_patches = (int)patches.size();
    if (n_patches == 0)
        return {};

    if (vox_env_bool_default_on("VOXCPM2_USE_GRAPH")) {
        return vae_decode_graph(ctx, patches);
    }

    int feat_dim = 64;
    int P = (int)ctx->hp.patch_frames; // 4

    // Total latent time frames = n_patches * P
    int T_lat = n_patches * P;

    // Upsampling ratios for layers 2-7: product = 8*6*5*2*2*2 = 1920
    // 25 Hz (latent rate) * 1920 = 48000 Hz output
    static const int up_rates[] = {8, 6, 5, 2, 2, 2};
    static const int n_up_blocks = 6;

    // Channel progression after each upsample block: 2048->1024->512->256->128->64->32
    static const int block_out_ch[] = {1024, 512, 256, 128, 64, 32};

    const auto& T = ctx->tensors;

    // Check if VAE weights exist -- look for the first input conv weight
    bool have_vae = (vae_tensor_f32(T, "vae.dec.layer.0.weight_g") != nullptr);

    if (!have_vae) {
        // Graceful degradation: return silence of computed duration
        // 25 Hz latent rate x 1920 upsample = 48000 Hz; P frames/patch -> P*1920 samples/patch
        std::vector<float> pcm((size_t)n_patches * (size_t)P * 1920, 0.0f);
        return pcm;
    }

    // --- Build latent tensor [feat_dim=64, T_lat] ---
    // Patches are [P * feat_dim] row-major = P frames of feat_dim each.
    // Latent tensor is [feat_dim, T_lat] = channels-first (C, T).
    std::vector<float> latents((size_t)feat_dim * T_lat, 0.0f);
    for (int n = 0; n < n_patches; n++) {
        const auto& patch = patches[n];
        for (int p = 0; p < P; p++) {
            int t = n * P + p;
            size_t patch_off = (size_t)p * feat_dim;
            if (patch_off >= patch.size())
                break;
            const float* src = patch.data() + patch_off;
            int avail = (int)std::min((size_t)feat_dim, patch.size() - patch_off);
            for (int c = 0; c < avail; c++) {
                latents[(size_t)c * T_lat + t] = src[c];
            }
        }
    }

    if (ctx->verbosity >= 2) {
        float mx = 0;
        for (auto v : latents) {
            float a = v < 0 ? -v : v;
            if (a > mx)
                mx = a;
        }
        fprintf(stderr, "voxcpm2 VAE: latent [%d, %d] max=%.4f\n", feat_dim, T_lat, mx);
    }

    // SR conditioning bucket for 48kHz
    int sr_bucket = 3;

    // --- Layer 0: depthwise in-conv (k=7, groups=64) ---
    // GGUF weight_v: [k=7, in_per_grp=1, out=64]  weight_g: [64]
    int Tc = T_lat;
    int Cc = feat_dim;
    std::vector<float> h;

    const bool vae_trace = vox_env_bool("VOXCPM2_VAE_TRACE");
    auto trace_dump = [&](const char* name, const std::vector<float>& v, int Cv, int Tv) {
        if (!vae_trace)
            return;
        float mx = 0.0f, sq = 0.0f;
        for (size_t i = 0; i < v.size(); i++) {
            float a = std::abs(v[i]);
            if (a > mx)
                mx = a;
            sq += v[i] * v[i];
        }
        float rms = std::sqrt(sq / (float)v.size());
        fprintf(stderr, "voxcpm2 VAE-trace [legacy] %-24s [C=%d, T=%d] n=%zu max=%.4f rms=%.4f\n", name, Cv, Tv,
                v.size(), mx, rms);
        std::string path = std::string("/tmp/voxcpm2_l_") + name + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f) {
            std::fwrite(v.data(), sizeof(float), v.size(), f);
            std::fclose(f);
        }
    };

    {
        const float* g0 = vae_tensor_f32(T, "vae.dec.layer.0.weight_g");
        const float* v0 = vae_tensor_f32(T, "vae.dec.layer.0.weight_v");
        const float* b0 = vae_tensor_f32(T, "vae.dec.layer.0.bias");
        if (g0 && v0) {
            // Depthwise: groups=feat_dim, in_per_grp=1
            const auto& w0 = vae_wn_get_cached(ctx, "vae.dec.layer.0", g0, v0, feat_dim, 1, 7);
            std::vector<float> h0((size_t)feat_dim * Tc, 0.0f);
            causal_conv1d(w0.data(), b0, latents.data(), h0.data(), feat_dim, feat_dim, 7, Tc, 1, 1, feat_dim);
            h = std::move(h0);
        } else {
            h = latents;
        }
    }
    trace_dump("after_layer0", h, Cc, Tc);
    if (ctx->verbosity >= 2) {
        float mx = 0;
        for (auto v : h) {
            float a = v < 0 ? -v : v;
            if (a > mx)
                mx = a;
        }
        fprintf(stderr, "voxcpm2 VAE: after layer0 (depthwise k=7): Cc=%d Tc=%d max=%.4f\n", Cc, Tc, mx);
    }

    // --- Layer 1: pointwise in-conv (k=1, 64->2048) ---
    {
        const float* g1 = vae_tensor_f32(T, "vae.dec.layer.1.weight_g");
        const float* v1 = vae_tensor_f32(T, "vae.dec.layer.1.weight_v");
        const float* b1 = vae_tensor_f32(T, "vae.dec.layer.1.bias");
        if (g1 && v1) {
            // weight_g is [out_ch] stored with varying GGUF shapes.
            // Total elements = out_ch regardless of ne[] layout.
            auto it_g1 = T.find("vae.dec.layer.1.weight_g");
            int out_ch1 = it_g1 != T.end() ? (int)ggml_nelements(it_g1->second) : 2048;
            const auto& w1 = vae_wn_get_cached(ctx, "vae.dec.layer.1", g1, v1, out_ch1, feat_dim, 1);
            std::vector<float> h1((size_t)out_ch1 * Tc, 0.0f);
            causal_conv1d(w1.data(), b1, h.data(), h1.data(), out_ch1, feat_dim, 1, Tc, 1, 1, 1);
            h = std::move(h1);
            Cc = out_ch1;
        }
    }
    if (ctx->verbosity >= 2) {
        float mx = 0;
        for (auto v : h) {
            float a = v < 0 ? -v : v;
            if (a > mx)
                mx = a;
        }
        fprintf(stderr, "voxcpm2 VAE: after layer1 (1x1 64->%d): Tc=%d max=%.4f\n", Cc, Tc, mx);
    }
    trace_dump("after_layer1", h, Cc, Tc);

    // --- Layers 2-7: upsample blocks ---
    const bool bench_vae = vox_env_bool("VOXCPM2_BENCH");
    for (int b = 0; b < n_up_blocks; b++) {
        int layer_idx = b + 2; // layers 2 through 7
        int up = up_rates[b];
        std::string lp = "vae.dec.layer." + std::to_string(layer_idx);
        double t_block0 = bench_vae ? vox_now_ms() : 0;
        double t_block_up = 0, t_block_res = 0;

        // SR conditioning: scale_embed and bias_embed are [channels, 4]
        // GGUF layout [channels, 4] -> ne[0]=4, ne[1]=channels
        // scale_embed[c, bucket] = data[bucket + c*4]
        // Apply: x[c, t] = x[c, t] * scale[c] + bias[c]
        {
            std::string sr_pfx = "vae.dec.sr_cond." + std::to_string(layer_idx);
            const float* se = vae_tensor_f32(T, sr_pfx + ".scale_embed");
            const float* be = vae_tensor_f32(T, sr_pfx + ".bias_embed");
            if (se) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int c = 0; c < Cc; c++) {
                    float sc = se[(size_t)c * 4 + sr_bucket];
                    float bi = be ? be[(size_t)c * 4 + sr_bucket] : 0.0f;
                    float* hc = h.data() + (size_t)c * Tc;
                    for (int t = 0; t < Tc; t++) {
                        hc[t] = hc[t] * sc + bi;
                    }
                }
            }
        }
        if (b == 0)
            trace_dump("block_0_sr", h, Cc, Tc);

        // Snake1d before upsample: .block.0.alpha
        {
            const float* alpha_up = vae_tensor_f32(T, lp + ".block.0.alpha");
            if (alpha_up) {
                snake1d(alpha_up, h.data(), h.data(), Cc, Tc);
            }
        }
        if (b == 0)
            trace_dump("block_0_snake", h, Cc, Tc);

        // CausalTransposeConv1d upsample: .block.1.{weight_g, weight_v, bias}
        // Python's CausalTransposeConv1d emits T_in * stride samples
        // (see causal_transposed_conv1d comment for derivation).
        int out_ch_b = block_out_ch[b];
        int T_up = Tc * up;
        std::vector<float> h_up((size_t)out_ch_b * T_up, 0.0f);
        {
            const float* g_up = vae_tensor_f32(T, lp + ".block.1.weight_g");
            const float* v_up = vae_tensor_f32(T, lp + ".block.1.weight_v");
            const float* b_up = vae_tensor_f32(T, lp + ".block.1.bias");
            if (g_up && v_up) {
                // GGUF transposed conv weight_v: [k, out_ch, in_ch]
                // ne[0]=k, ne[1]=out_ch, ne[2]=in_ch
                // ksize from ne[0] of weight_v
                int ksize_up = vae_tensor_dim(T, lp + ".block.1.weight_v", 0);
                if (ksize_up <= 0)
                    ksize_up = 2 * up;
                // wn_reconstruct(g, v, out_ch=Cc, in_ch=out_ch_b, k)
                // treats weight_v as [k, in_ch=out_ch_b, out_ch=Cc]
                // output layout [Cc, out_ch_b, k] = [in_ch, out_ch, k] for causal_transposed_conv1d
                const auto& w_up = vae_wn_get_cached(ctx, lp + ".block.1", g_up, v_up, Cc, out_ch_b, ksize_up);
                causal_transposed_conv1d(w_up.data(), b_up, h.data(), h_up.data(), Cc, out_ch_b, ksize_up, Tc, up);
            } else {
                // Repeat-interpolate fallback (only when weights missing).
                int copy_ch = std::min(Cc, out_ch_b);
                for (int c = 0; c < copy_ch; c++) {
                    for (int t = 0; t < Tc; t++) {
                        float val = h[(size_t)c * Tc + t];
                        for (int u = 0; u < up; u++) {
                            h_up[(size_t)c * T_up + t * up + u] = val;
                        }
                    }
                }
            }
        }

        Tc = T_up;
        Cc = out_ch_b;
        h = std::move(h_up);

        if (bench_vae) {
            t_block_up = vox_now_ms() - t_block0;
        }
        if (ctx->verbosity >= 2) {
            float mx = 0;
            for (size_t i = 0; i < (size_t)Cc * Tc; i++) {
                float a = h[i] < 0 ? -h[i] : h[i];
                if (a > mx)
                    mx = a;
            }
            fprintf(stderr, "voxcpm2 VAE: block %d upsample(%d): Cc=%d Tc=%d max=%.4f\n", b, up, Cc, Tc, mx);
        }
        {
            std::string nm = "block_" + std::to_string(b) + "_upsample";
            trace_dump(nm.c_str(), h, Cc, Tc);
        }

        double t_res0 = bench_vae ? vox_now_ms() : 0;
        // 3x CausalResidualUnit: .block.{2,3,4} with dilations 1, 3, 9
        for (int r = 0; r < 3; r++) {
            int dil = (r == 0) ? 1 : (r == 1) ? 3 : 9;
            std::string rp = lp + ".block." + std::to_string(r + 2);
            std::vector<float> h_res((size_t)Cc * Tc);
            vae_residual_unit(ctx, rp, h.data(), h_res.data(), Cc, Tc, dil);
            h = std::move(h_res);
        }
        if (bench_vae) {
            t_block_res = vox_now_ms() - t_res0;
            fprintf(stderr, "voxcpm2[bench]:   vae block %d (up=%d Cc=%d Tc=%d): upsample=%7.1f ms res=%7.1f ms\n", b,
                    up, Cc, Tc, t_block_up, t_block_res);
        }
        {
            std::string nm = "block_" + std::to_string(b) + "_residuals";
            trace_dump(nm.c_str(), h, Cc, Tc);
        }
    }

    if (ctx->verbosity >= 2) {
        float mx = 0;
        for (size_t i = 0; i < (size_t)Cc * Tc; i++) {
            float a = h[i] < 0 ? -h[i] : h[i];
            if (a > mx)
                mx = a;
        }
        fprintf(stderr, "voxcpm2 VAE: after upsample blocks: Cc=%d Tc=%d max=%.4f\n", Cc, Tc, mx);
    }

    // --- Layer 8: final Snake1d ---
    {
        const float* alpha_f = vae_tensor_f32(T, "vae.dec.layer.8.alpha");
        if (alpha_f) {
            snake1d(alpha_f, h.data(), h.data(), Cc, Tc);
        }
    }

    // --- Layer 9: final out-conv (k=7, Cc->1) then Tanh ---
    std::vector<float> pcm(Tc, 0.0f);
    {
        const float* g9 = vae_tensor_f32(T, "vae.dec.layer.9.weight_g");
        const float* v9 = vae_tensor_f32(T, "vae.dec.layer.9.weight_v");
        const float* b9 = vae_tensor_f32(T, "vae.dec.layer.9.bias");
        if (g9 && v9) {
            const auto& w9 = vae_wn_get_cached(ctx, "vae.dec.layer.9", g9, v9, 1, Cc, 7);
            causal_conv1d(w9.data(), b9, h.data(), pcm.data(), 1, Cc, 7, Tc, 1, 1, 1);
            for (float& s : pcm)
                s = std::tanh(s);
        } else {
            // Mix-down fallback
            float inv_Cc = 1.0f / std::max(1, Cc);
            for (int t = 0; t < Tc; t++) {
                float mix = 0.0f;
                for (int c = 0; c < Cc; c++)
                    mix += h[(size_t)c * Tc + t];
                pcm[t] = std::tanh(mix * inv_Cc);
            }
        }
    }

    return pcm;
}

// ===========================================================================
// VAE encoder — used by voice cloning. Encodes 16 kHz mono PCM into latent
// patches [T_patches, P=4, D=64] for the reference prefix in TSLM prefill.
//
// Architecture (mirrors audio_vae_v2.py CausalEncoder):
//   conv0:        WNCausalConv1d(1 → vae_enc_dim, k=7, p=3)
//   blk.{0..3}:   3× CausalResidualUnit (depthwise, dilation 1/3/9)
//                 + Snake1d
//                 + WNCausalConv1d(C → 2C, k=2s, stride=s, p=ceil(s/2),
//                                  output_padding=s%2)   ← dense, NOT depthwise
//   fc_mu:        WNCausalConv1d(C_last → latent_dim, k=3, p=1)
//
// We use the existing `causal_conv1d` for everything except the strided
// downsamples, which need a different effective padding (2*p - op rather
// than k-1) — that's `vae_strided_conv1d` below.
// ===========================================================================

// CausalConv1d with explicit (left) pad — matches PyTorch's
// F.pad(x, (2*p - op, 0)); Conv1d(stride=s, padding=0). Used for the encoder's
// strided downsamples where Python's `2p-op` is smaller than `(k-1)*dilation`.
// weight layout: [out_ch, in_ch, ksize] (post-wn_reconstruct).
static void vae_strided_conv1d(const float* weight, const float* bias, const float* x_in, float* x_out, int in_ch,
                               int out_ch, int ksize, int T_in, int stride, int left_pad) {
    int T_out = (T_in + left_pad - ksize) / stride + 1;
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ot = 0; ot < T_out; ot++) {
            float b_val = bias ? bias[oc] : 0.0f;
            float acc = b_val;
            int it_start = ot * stride - left_pad;
            for (int k = 0; k < ksize; k++) {
                int it = it_start + k;
                if (it < 0 || it >= T_in)
                    continue;
                for (int ic = 0; ic < in_ch; ic++) {
                    float xv = x_in[(size_t)ic * T_in + it];
                    float wv = weight[(size_t)oc * in_ch * ksize + (size_t)ic * ksize + k];
                    acc += xv * wv;
                }
            }
            x_out[(size_t)oc * T_out + ot] = acc;
        }
    }
}

// One encoder block (3 ResUnits + Snake + strided downsample).
// Returns the new time length T_out (= (T_in + 2*ceil(s/2) - s%2 - 2*s) / s + 1
// which simplifies to T_in / s for all the rates we use).
static int vae_enc_block(voxcpm2_context* ctx, int blk_idx, int in_ch, int out_ch, int stride, const float* x_in,
                         std::vector<float>& x_out, int T_in) {
    const auto& T = ctx->tensors;
    std::string blk = "vae.enc.blk." + std::to_string(blk_idx);

    // 3 residual units (depthwise, dilation 1, 3, 9) — reuses vae_residual_unit
    std::vector<float> h0((size_t)in_ch * T_in);
    vae_residual_unit(ctx, blk + ".res.0", x_in, h0.data(), in_ch, T_in, 1);
    std::vector<float> h1((size_t)in_ch * T_in);
    vae_residual_unit(ctx, blk + ".res.1", h0.data(), h1.data(), in_ch, T_in, 3);
    std::vector<float> h2((size_t)in_ch * T_in);
    vae_residual_unit(ctx, blk + ".res.2", h1.data(), h2.data(), in_ch, T_in, 9);

    // Snake1d before the strided downsample
    std::vector<float> h3((size_t)in_ch * T_in);
    const float* alpha_d = vae_tensor_f32(T, blk + ".sub.3.alpha");
    if (alpha_d) {
        snake1d(alpha_d, h2.data(), h3.data(), in_ch, T_in);
    } else {
        std::memcpy(h3.data(), h2.data(), (size_t)in_ch * T_in * sizeof(float));
    }

    // Strided downsample (dense, groups=1). Python:
    //   k = 2*stride; p = ceil(s/2); op = s%2; left_pad = 2p - op
    int ksize = 2 * stride;
    int python_pad = 2 * ((stride + 1) / 2) - (stride % 2);
    int T_out = (T_in + python_pad - ksize) / stride + 1;
    if (T_out <= 0) {
        x_out.clear();
        return 0;
    }
    x_out.assign((size_t)out_ch * T_out, 0.0f);

    const float* g = vae_tensor_f32(T, blk + ".sub.4.weight_g");
    const float* v = vae_tensor_f32(T, blk + ".sub.4.weight_v");
    const float* b = vae_tensor_f32(T, blk + ".sub.4.bias");
    if (g && v) {
        const auto& w = vae_wn_get_cached(ctx, blk + ".sub.4", g, v, out_ch, in_ch, ksize);
        vae_strided_conv1d(w.data(), b, h3.data(), x_out.data(), in_ch, out_ch, ksize, T_in, stride, python_pad);
    }
    return T_out;
}

// Top-level VAE encoder. Takes raw 16 kHz mono float32 PCM (`pcm`,
// `n_samples`), returns latent patches packed as [T_patches, P, D] row-major
// (matching Python `feat.view(D, -1, P).permute(1, 2, 0)`).
// On failure (encoder weights missing) returns an empty vector and sets
// *out_T_patches = 0.
// Process-wide VAE encode cache. The diff harness calls extract_stage once per
// stage and several stages need ref_feat — the encoder takes ~30 s on CPU so
// re-running it 5-10 times serialises into multiple minutes. Cache keyed on
// (ctx, ref pointer, ref length).
struct vox_vae_cache_key {
    voxcpm2_context* ctx;
    const float* pcm;
    int n_samples;
    bool operator==(const vox_vae_cache_key& o) const {
        return ctx == o.ctx && pcm == o.pcm && n_samples == o.n_samples;
    }
};

static std::vector<float> vae_encode_uncached(voxcpm2_context* ctx, const float* pcm, int n_samples,
                                              int* out_T_patches);

static const std::vector<float>& vae_encode_cached(voxcpm2_context* ctx, const float* pcm, int n_samples,
                                                   int* out_T_patches) {
    static vox_vae_cache_key cached_key{nullptr, nullptr, -1};
    static std::vector<float> cached;
    static int cached_T = 0;
    vox_vae_cache_key key{ctx, pcm, n_samples};
    if (!(key == cached_key)) {
        cached = vae_encode_uncached(ctx, pcm, n_samples, &cached_T);
        cached_key = key;
    }
    if (out_T_patches) {
        *out_T_patches = cached_T;
    }
    return cached;
}

static std::vector<float> vae_encode(voxcpm2_context* ctx, const float* pcm, int n_samples, int* out_T_patches) {
    return vae_encode_uncached(ctx, pcm, n_samples, out_T_patches);
}

static std::vector<float> vae_encode_uncached(voxcpm2_context* ctx, const float* pcm, int n_samples,
                                              int* out_T_patches) {
    const auto& T = ctx->tensors;
    const auto& hp = ctx->hp;

    if (out_T_patches) {
        *out_T_patches = 0;
    }

    int P = (int)hp.patch_frames;                // 4
    int latent_dim = (int)hp.vae_enc_latent_dim; // 64
    int d_model = (int)hp.vae_enc_dim;           // 128
    int n_blocks = (int)hp.vae_enc_n_blocks;     // 4

    // Encoder hop length = product of strides; pad input to a multiple of
    // patch_len = P * hop so that T_lat % P == 0.
    int hop = 1;
    for (int b = 0; b < n_blocks; b++) {
        hop *= (int)hp.vae_enc_rates[b];
    }
    int patch_len = P * hop;
    if (n_samples <= 0 || patch_len <= 0) {
        return {};
    }
    int padded_n = ((n_samples + patch_len - 1) / patch_len) * patch_len;
    std::vector<float> x((size_t)padded_n, 0.0f);
    std::memcpy(x.data(), pcm, (size_t)n_samples * sizeof(float));

    // --- conv0: depthwise=False, in=1, out=d_model, k=7, p=3 (2p-op = k-1) ---
    int ksize = 7;
    std::vector<float> h((size_t)d_model * padded_n);
    {
        const float* g = vae_tensor_f32(T, "vae.enc.conv0.weight_g");
        const float* v = vae_tensor_f32(T, "vae.enc.conv0.weight_v");
        const float* b = vae_tensor_f32(T, "vae.enc.conv0.bias");
        if (!g || !v) {
            fprintf(stderr, "voxcpm2: VAE encoder weights missing (vae.enc.conv0.*) — cannot voice-clone\n");
            return {};
        }
        const auto& w = vae_wn_get_cached(ctx, "vae.enc.conv0", g, v, d_model, /*in_ch=*/1, ksize);
        causal_conv1d(w.data(), b, x.data(), h.data(), d_model, /*in_ch=*/1, ksize, padded_n,
                      /*stride=*/1, /*dilation=*/1, /*groups=*/1);
    }

    // --- 4 encoder blocks: channels double, time divides by stride ---
    int cur_C = d_model;
    int cur_T = padded_n;
    std::vector<float> cur = std::move(h);
    for (int b = 0; b < n_blocks; b++) {
        int stride = (int)hp.vae_enc_rates[b];
        int next_C = cur_C * 2;
        std::vector<float> next;
        cur_T = vae_enc_block(ctx, b, cur_C, next_C, stride, cur.data(), next, cur_T);
        cur = std::move(next);
        cur_C = next_C;
        if (cur_T <= 0) {
            return {};
        }
    }

    // --- fc_mu: [cur_C → latent_dim, k=3, p=1]  (2p-op = k-1, reuse causal_conv1d) ---
    ksize = 3;
    std::vector<float> mu_out((size_t)latent_dim * cur_T);
    {
        const float* g = vae_tensor_f32(T, "vae.enc.fc_mu.weight_g");
        const float* v = vae_tensor_f32(T, "vae.enc.fc_mu.weight_v");
        const float* b = vae_tensor_f32(T, "vae.enc.fc_mu.bias");
        if (!g || !v) {
            fprintf(stderr, "voxcpm2: VAE encoder weights missing (vae.enc.fc_mu.*) — cannot voice-clone\n");
            return {};
        }
        const auto& w = vae_wn_get_cached(ctx, "vae.enc.fc_mu", g, v, latent_dim, cur_C, ksize);
        causal_conv1d(w.data(), b, cur.data(), mu_out.data(), latent_dim, cur_C, ksize, cur_T,
                      /*stride=*/1, /*dilation=*/1, /*groups=*/1);
    }

    // --- Reshape [latent_dim=64, T_lat] → [T_patches, P, latent_dim] ---
    int T_patches = cur_T / P;
    if (out_T_patches) {
        *out_T_patches = T_patches;
    }
    std::vector<float> out((size_t)T_patches * P * latent_dim);
    for (int tp = 0; tp < T_patches; tp++) {
        for (int pf = 0; pf < P; pf++) {
            int t = tp * P + pf;
            for (int d = 0; d < latent_dim; d++) {
                out[((size_t)tp * P + pf) * latent_dim + d] = mu_out[(size_t)d * cur_T + t];
            }
        }
    }
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: VAE encoded %d samples → %d patches (P=%d, D=%d)\n", n_samples, T_patches, P,
                latent_dim);
    }
    return out;
}

// ---------------------------------------------------------------------------
// GPT-2 byte encoder table (built lazily)
// ---------------------------------------------------------------------------

static const std::vector<int>& vox_byte_encoder() {
    static std::vector<int> bs(256, 0);
    static bool init = false;
    if (init)
        return bs;
    std::vector<int> printable;
    for (int b = 0x21; b <= 0x7e; b++)
        printable.push_back(b);
    for (int b = 0xa1; b <= 0xac; b++)
        printable.push_back(b);
    for (int b = 0xae; b <= 0xff; b++)
        printable.push_back(b);
    int next_extra = 256;
    for (int b = 0; b < 256; b++) {
        bool is_p = false;
        for (int p : printable)
            if (p == b) {
                is_p = true;
                break;
            }
        bs[b] = is_p ? b : next_extra++;
    }
    init = true;
    return bs;
}

static void utf8_encode_cp(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

static std::string vox_bytes_to_unicode(const char* bytes, size_t n) {
    auto& enc = vox_byte_encoder();
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        utf8_encode_cp((uint32_t)enc[(unsigned char)bytes[i]], out);
    }
    return out;
}

// BPE merge for one byte-encoded pre-token
static void vox_bpe_one(const vox_tokenizer& tok, const std::string& word, std::vector<int32_t>& out) {
    if (word.empty())
        return;

    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < word.size()) {
        unsigned char c = (unsigned char)word[i];
        size_t len;
        if (c < 0x80)
            len = 1;
        else if ((c & 0xE0) == 0xC0)
            len = 2;
        else if ((c & 0xF0) == 0xE0)
            len = 3;
        else if ((c & 0xF8) == 0xF0)
            len = 4;
        else
            len = 1;
        if (i + len > word.size())
            len = 1;
        symbols.emplace_back(word, i, len);
        i += len;
    }

    if (!tok.merge_rank.empty()) {
        int max_iter = (int)symbols.size();
        for (int iter = 0; iter < max_iter && symbols.size() >= 2; iter++) {
            int best_i = -1, best_rank = INT_MAX;
            for (size_t k = 0; k + 1 < symbols.size(); k++) {
                std::string pair = symbols[k] + " " + symbols[k + 1];
                auto it = tok.merge_rank.find(pair);
                if (it != tok.merge_rank.end() && (int)it->second < best_rank) {
                    best_rank = (int)it->second;
                    best_i = (int)k;
                }
            }
            if (best_i < 0)
                break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }
    }

    for (const auto& s : symbols) {
        auto it = tok.token_to_id.find(s);
        if (it != tok.token_to_id.end()) {
            out.push_back(it->second);
        } else {
            // Per-byte fallback
            size_t j = 0;
            while (j < s.size()) {
                unsigned char c2 = (unsigned char)s[j];
                size_t len;
                if (c2 < 0x80)
                    len = 1;
                else if ((c2 & 0xE0) == 0xC0)
                    len = 2;
                else if ((c2 & 0xF0) == 0xE0)
                    len = 3;
                else if ((c2 & 0xF8) == 0xF0)
                    len = 4;
                else
                    len = 1;
                auto jt = tok.token_to_id.find(std::string(s, j, len));
                if (jt != tok.token_to_id.end())
                    out.push_back(jt->second);
                j += len;
            }
        }
    }
}

// CJK codepoint check for post-expansion
static bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0xF900 && cp <= 0xFAFF);
}

// Decode one UTF-8 codepoint from s at position i, advance i
static uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        i += 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        uint32_t cp = ((c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
        i += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        uint32_t cp = ((c & 0x0F) << 12) | (((unsigned char)s[i + 1] & 0x3F) << 6) | ((unsigned char)s[i + 2] & 0x3F);
        i += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        uint32_t cp = ((c & 0x07) << 18) | (((unsigned char)s[i + 1] & 0x3F) << 12) |
                      (((unsigned char)s[i + 2] & 0x3F) << 6) | ((unsigned char)s[i + 3] & 0x3F);
        i += 4;
        return cp;
    }
    i += 1;
    return 0xFFFD;
}

// Tokenize: SentencePiece BPE with ▁ word boundaries + CJK post-expansion
static std::vector<int32_t> vox_tokenize(const vox_tokenizer& tok, const std::string& text) {
    // Step 1: Normalize — prepend ▁, replace spaces with ▁
    // ▁ = U+2581 = \xe2\x96\x81
    std::string normalized = "\xe2\x96\x81";
    for (char c : text) {
        if (c == ' ')
            normalized += "\xe2\x96\x81";
        else
            normalized += c;
    }

    // Step 2: BPE encode using vocab ordering as merge priority
    // Start with UTF-8 codepoint symbols
    std::vector<std::string> symbols;
    {
        size_t pos = 0;
        while (pos < normalized.size()) {
            size_t prev = pos;
            utf8_next(normalized, pos);
            symbols.push_back(normalized.substr(prev, pos - prev));
        }
    }

    // Greedy BPE merge: repeatedly merge the pair with lowest merged-token ID
    while (symbols.size() > 1) {
        int best_id = INT_MAX, best_pos = -1;
        for (int k = 0; k + 1 < (int)symbols.size(); k++) {
            std::string merged = symbols[k] + symbols[k + 1];
            auto it = tok.token_to_id.find(merged);
            if (it != tok.token_to_id.end() && it->second < best_id) {
                best_id = it->second;
                best_pos = k;
            }
        }
        if (best_pos < 0)
            break;
        symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    // Convert symbols to IDs
    std::vector<int32_t> result;
    for (const auto& sym : symbols) {
        auto it = tok.token_to_id.find(sym);
        if (it != tok.token_to_id.end()) {
            result.push_back(it->second);
        } else {
            // Byte fallback: encode each byte as <0xNN>
            for (unsigned char c : sym) {
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", c);
                auto jt = tok.token_to_id.find(hex);
                if (jt != tok.token_to_id.end())
                    result.push_back(jt->second);
            }
        }
    }

    // Step 3: CJK post-expansion — split multi-char CJK tokens into individual chars
    std::vector<int32_t> expanded;
    for (int32_t id : result) {
        if (id < 0 || id >= (int32_t)tok.id_to_token.size()) {
            expanded.push_back(id);
            continue;
        }
        const std::string& ts = tok.id_to_token[id];
        // Check if token is multi-char CJK (remove ▁ prefix first)
        std::string clean = ts;
        while (clean.size() >= 3 && clean.substr(0, 3) == "\xe2\x96\x81")
            clean = clean.substr(3);
        if (clean.empty()) {
            expanded.push_back(id);
            continue;
        }
        // Count CJK codepoints
        int n_cjk = 0, n_total = 0;
        {
            size_t p = 0;
            while (p < clean.size()) {
                if (is_cjk(utf8_next(clean, p)))
                    n_cjk++;
                n_total++;
            }
        }
        if (n_cjk == n_total && n_total >= 2) {
            // Split into individual chars
            size_t p = 0;
            while (p < clean.size()) {
                size_t prev = p;
                utf8_next(clean, p);
                std::string ch = clean.substr(prev, p - prev);
                auto jt = tok.token_to_id.find(ch);
                if (jt != tok.token_to_id.end())
                    expanded.push_back(jt->second);
                else
                    expanded.push_back(id); // fallback: keep original
            }
        } else {
            expanded.push_back(id);
        }
    }
    result = expanded;

    if (tok.audio_start_token >= 0) {
        result.push_back(tok.audio_start_token);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Model loading — two-pass GGUF
// ---------------------------------------------------------------------------

static bool vox_load_weights(voxcpm2_context* ctx, const char* path) {
    using namespace core_gguf;

    // --- Pass 1: metadata ---
    gguf_context* meta = open_metadata(path);
    if (!meta)
        return false;

    vox_hparams& hp = ctx->hp;

    // GGUF keys use "voxcpm2." prefix (set by the converter)
    hp.tslm_n_layers = kv_u32(meta, "voxcpm2.tslm.n_layers", hp.tslm_n_layers);
    hp.tslm_d_model = kv_u32(meta, "voxcpm2.tslm.d_model", hp.tslm_d_model);
    hp.tslm_n_heads = kv_u32(meta, "voxcpm2.tslm.n_heads", hp.tslm_n_heads);
    hp.tslm_n_kv = kv_u32(meta, "voxcpm2.tslm.n_kv_heads", hp.tslm_n_kv);
    hp.tslm_head_dim = kv_u32(meta, "voxcpm2.tslm.head_dim", hp.tslm_head_dim);
    hp.tslm_ff_dim = kv_u32(meta, "voxcpm2.tslm.ff_dim", hp.tslm_ff_dim);
    hp.tslm_max_pos = kv_u32(meta, "voxcpm2.tslm.max_pos", hp.tslm_max_pos);
    hp.tslm_rope_theta = kv_f32(meta, "voxcpm2.tslm.rope_theta", hp.tslm_rope_theta);
    hp.rms_norm_eps = kv_f32(meta, "voxcpm2.tslm.rms_norm_eps", hp.rms_norm_eps);

    hp.ralm_n_layers = kv_u32(meta, "voxcpm2.ralm.n_layers", hp.ralm_n_layers);
    hp.ralm_d_model = kv_u32(meta, "voxcpm2.ralm.d_model", hp.ralm_d_model);
    hp.ralm_n_heads = kv_u32(meta, "voxcpm2.ralm.n_heads", hp.ralm_n_heads);
    hp.ralm_n_kv = kv_u32(meta, "voxcpm2.ralm.n_kv_heads", hp.ralm_n_kv);
    hp.ralm_head_dim = kv_u32(meta, "voxcpm2.ralm.head_dim", hp.ralm_head_dim);
    hp.ralm_ff_dim = kv_u32(meta, "voxcpm2.ralm.ff_dim", hp.ralm_ff_dim);

    hp.locenc_n_layers = kv_u32(meta, "voxcpm2.locenc.n_layers", hp.locenc_n_layers);
    hp.locenc_d_model = kv_u32(meta, "voxcpm2.locenc.d_model", hp.locenc_d_model);
    hp.locenc_n_heads = kv_u32(meta, "voxcpm2.locenc.n_heads", hp.locenc_n_heads);
    hp.locenc_n_kv = kv_u32(meta, "voxcpm2.locenc.n_kv_heads", hp.locenc_n_kv);
    hp.locenc_head_dim = kv_u32(meta, "voxcpm2.locenc.head_dim", hp.locenc_head_dim);
    hp.locenc_ff_dim = kv_u32(meta, "voxcpm2.locenc.ff_dim", hp.locenc_ff_dim);

    hp.locdit_n_layers = kv_u32(meta, "voxcpm2.locdit.n_layers", hp.locdit_n_layers);
    hp.locdit_d_model = kv_u32(meta, "voxcpm2.locdit.d_model", hp.locdit_d_model);
    hp.locdit_n_heads = kv_u32(meta, "voxcpm2.locdit.n_heads", hp.locdit_n_heads);
    hp.locdit_n_kv = kv_u32(meta, "voxcpm2.locdit.n_kv_heads", hp.locdit_n_kv);
    hp.locdit_head_dim = kv_u32(meta, "voxcpm2.locdit.head_dim", hp.locdit_head_dim);
    hp.locdit_ff_dim = kv_u32(meta, "voxcpm2.locdit.ff_dim", hp.locdit_ff_dim);

    // Token IDs
    hp.n_vocab = kv_u32(meta, "tokenizer.n_vocab", hp.n_vocab);
    hp.audio_start_token = kv_u32(meta, "voxcpm2.audio_start_token", hp.audio_start_token);
    hp.audio_end_token = kv_u32(meta, "voxcpm2.audio_end_token", hp.audio_end_token);
    hp.ref_audio_start_token = kv_u32(meta, "voxcpm2.ref_audio_start_token", hp.ref_audio_start_token);
    hp.ref_audio_end_token = kv_u32(meta, "voxcpm2.ref_audio_end_token", hp.ref_audio_end_token);

    hp.patch_frames = kv_u32(meta, "voxcpm2.patch_frames", hp.patch_frames);
    hp.patch_dim = kv_u32(meta, "voxcpm2.vae.patch_dim", hp.patch_dim);

    // VAE encoder dimensions (architectural constants; default to AudioVAEConfig
    // defaults if not specified in GGUF metadata).
    hp.vae_enc_dim = kv_u32(meta, "voxcpm2.vae.encoder_dim", hp.vae_enc_dim);
    hp.vae_enc_latent_dim = kv_u32(meta, "voxcpm2.vae.latent_dim", hp.vae_enc_latent_dim);
    hp.vae_enc_sample_rate = kv_u32(meta, "voxcpm2.vae.sample_rate", hp.vae_enc_sample_rate);

    // Tokenizer: try GGUF string arrays first, then vocab blob tensor
    {
        auto tokens = kv_str_array(meta, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            ctx->tokenizer.id_to_token = tokens;
            for (size_t b = 0; b < tokens.size(); b++)
                ctx->tokenizer.token_to_id[tokens[b]] = (int32_t)b;
            auto merges = kv_str_array(meta, "tokenizer.ggml.merges");
            for (size_t b = 0; b < merges.size(); b++)
                ctx->tokenizer.merge_rank[merges[b]] = (int32_t)b;
        }
        // Note: vocab blob tensor is loaded after weights (see below)
    }

    free_metadata(meta);

    // --- Pass 2: weights ---
    // Load onto a host-visible backend so legacy CPU paths can
    // dereference tensor->data. On unified-memory backends (Metal) this
    // is the GPU backend itself; on discrete GPUs (Vulkan, CUDA) it's
    // the CPU backend. GPU mirrors are created later by the caller.
    ggml_backend_t weight_backend = ctx->backend ? ctx->backend : get_cpu_backend();
    if (weight_backend && weight_backend != ctx->backend_cpu) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(weight_backend);
        if (buft && !ggml_backend_buft_is_host(buft)) {
            weight_backend = ctx->backend_cpu;
        }
    }
    WeightLoad wl;
    if (!load_weights(path, weight_backend, "voxcpm2", wl))
        return false;

    ctx->ggml_ctx = wl.ctx;
    ctx->weight_buf = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    auto& T = ctx->tensors;
    vox_weights& W = ctx->weights;

    // Infer n_kv for LocEnc/LocDiT from K weight shapes when not in metadata.
    // K weight: ne[0]=d_model (input), ne[1]=n_kv*head_dim (output)
    {
        auto it = T.find("ralm.blk.0.attn_k.weight");
        if (it != T.end() && hp.ralm_head_dim > 0) {
            uint32_t kv_dim = (uint32_t)it->second->ne[1];
            uint32_t inferred = kv_dim / hp.ralm_head_dim;
            if (inferred > 0 && inferred != hp.ralm_n_kv) {
                if (ctx->verbosity >= 1)
                    fprintf(stderr, "voxcpm2: ralm n_kv inferred from K weight: %u (was %u)\n", inferred, hp.ralm_n_kv);
                hp.ralm_n_kv = inferred;
            }
        }
    }
    {
        auto it = T.find("locenc.blk.0.attn_k.weight");
        if (it != T.end() && hp.locenc_head_dim > 0) {
            uint32_t kv_dim = (uint32_t)it->second->ne[1];
            uint32_t inferred = kv_dim / hp.locenc_head_dim;
            if (inferred > 0 && inferred != hp.locenc_n_kv) {
                if (ctx->verbosity >= 1)
                    fprintf(stderr, "voxcpm2: locenc n_kv inferred from K weight: %u (was %u)\n", inferred,
                            hp.locenc_n_kv);
                hp.locenc_n_kv = inferred;
            }
        }
    }
    {
        auto it = T.find("locdit.blk.0.attn_k.weight");
        if (it != T.end() && hp.locdit_head_dim > 0) {
            uint32_t kv_dim = (uint32_t)it->second->ne[1];
            uint32_t inferred = kv_dim / hp.locdit_head_dim;
            if (inferred > 0 && inferred != hp.locdit_n_kv) {
                if (ctx->verbosity >= 1)
                    fprintf(stderr, "voxcpm2: locdit n_kv inferred from K weight: %u (was %u)\n", inferred,
                            hp.locdit_n_kv);
                hp.locdit_n_kv = inferred;
            }
        }
    }

    // Check if vocab blob tensor exists and override tokenizer from it.
    // The GGUF stores each vocab entry as [uint16 len][bytes...] packed into
    // a 1-D F32 tensor where each float holds one byte value.
    {
        ggml_tensor* vocab_t = try_get(T, "tokenizer.vocab_tensor");
        if (vocab_t && ctx->tokenizer.id_to_token.empty()) {
            int n = (int)ggml_nelements(vocab_t);
            std::vector<float> fp_buf(n);
            ggml_backend_tensor_get(vocab_t, fp_buf.data(), 0, (size_t)n * sizeof(float));
            // Decode F32 -> uint8 raw bytes
            std::vector<uint8_t> raw(n);
            for (int i = 0; i < n; i++)
                raw[i] = (uint8_t)(int)fp_buf[i];
            // Parse [uint16 len][bytes len] entries
            ctx->tokenizer.id_to_token.clear();
            ctx->tokenizer.token_to_id.clear();
            int offset = 0;
            int n_vocab = (int)hp.n_vocab;
            for (int v = 0; v < n_vocab && offset + 2 <= n; v++) {
                uint16_t len = (uint16_t)(raw[offset] | (raw[offset + 1] << 8));
                offset += 2;
                if (offset + len > n)
                    break;
                std::string tok_str((const char*)(raw.data() + offset), len);
                offset += len;
                ctx->tokenizer.id_to_token.push_back(tok_str);
                ctx->tokenizer.token_to_id[tok_str] = (int32_t)v;
            }
            if (ctx->verbosity >= 1) {
                fprintf(stderr, "voxcpm2: loaded %zu tokens from vocab blob\n", ctx->tokenizer.id_to_token.size());
            }
        }
    }

    ctx->tokenizer.audio_start_token = (int32_t)hp.audio_start_token;

    // Override vocab size from GGUF metadata
    if (hp.n_vocab > 0 && ctx->tokenizer.id_to_token.size() < (size_t)hp.n_vocab) {
        ctx->tokenizer.id_to_token.resize(hp.n_vocab);
    }

    // Bind TSLM (note: tslm.lm_head.weight is NOT present in the GGUF)
    W.tslm_token_embd = require(T, "tslm.token_embd.weight", "voxcpm2");
    W.tslm_output_norm = require(T, "tslm.output_norm.weight", "voxcpm2");
    W.tslm_rope_short = try_get(T, "tslm.rope_short_factors");
    W.tslm_rope_long = try_get(T, "tslm.rope_long_factors");

    W.tslm_layers.resize(hp.tslm_n_layers);
    for (uint32_t i = 0; i < hp.tslm_n_layers; i++) {
        auto& L = W.tslm_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "tslm.blk.%u.%s", i, suffix);
            return nb;
        };
        L.attn_norm_w = require(T, fn("attn_norm.weight"), "voxcpm2");
        L.attn_q_w = require(T, fn("attn_q.weight"), "voxcpm2");
        L.attn_k_w = require(T, fn("attn_k.weight"), "voxcpm2");
        L.attn_v_w = require(T, fn("attn_v.weight"), "voxcpm2");
        L.attn_o_w = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_norm_w = require(T, fn("ffn_norm.weight"), "voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"), "voxcpm2");
        L.ffn_up_w = require(T, fn("ffn_up.weight"), "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"), "voxcpm2");
    }

    // Bind RALM
    W.ralm_output_norm = require(T, "ralm.output_norm.weight", "voxcpm2");
    W.ralm_layers.resize(hp.ralm_n_layers);
    for (uint32_t i = 0; i < hp.ralm_n_layers; i++) {
        auto& L = W.ralm_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "ralm.blk.%u.%s", i, suffix);
            return nb;
        };
        L.attn_norm_w = require(T, fn("attn_norm.weight"), "voxcpm2");
        L.attn_q_w = require(T, fn("attn_q.weight"), "voxcpm2");
        L.attn_k_w = require(T, fn("attn_k.weight"), "voxcpm2");
        L.attn_v_w = require(T, fn("attn_v.weight"), "voxcpm2");
        L.attn_o_w = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_norm_w = require(T, fn("ffn_norm.weight"), "voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"), "voxcpm2");
        L.ffn_up_w = require(T, fn("ffn_up.weight"), "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"), "voxcpm2");
    }

    // FSQ
    W.fsq_in_proj_w = require(T, "fsq.in_proj.weight", "voxcpm2");
    W.fsq_in_proj_b = require(T, "fsq.in_proj.bias", "voxcpm2");
    W.fsq_out_proj_w = require(T, "fsq.out_proj.weight", "voxcpm2");
    W.fsq_out_proj_b = require(T, "fsq.out_proj.bias", "voxcpm2");

    // LocEnc
    W.locenc_cls_token = try_get(T, "locenc.cls_token");
    W.locenc_in_proj_w = require(T, "locenc.in_proj.weight", "voxcpm2");
    W.locenc_in_proj_b = require(T, "locenc.in_proj.bias", "voxcpm2");
    W.locenc_norm_w = try_get(T, "locenc.output_norm.weight");
    W.locenc_layers.resize(hp.locenc_n_layers);
    for (uint32_t i = 0; i < hp.locenc_n_layers; i++) {
        auto& L = W.locenc_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "locenc.blk.%u.%s", i, suffix);
            return nb;
        };
        L.norm1_w = require(T, fn("attn_norm.weight"), "voxcpm2");
        L.norm2_w = require(T, fn("ffn_norm.weight"), "voxcpm2");
        L.attn_q_w = require(T, fn("attn_q.weight"), "voxcpm2");
        L.attn_k_w = require(T, fn("attn_k.weight"), "voxcpm2");
        L.attn_v_w = require(T, fn("attn_v.weight"), "voxcpm2");
        L.attn_o_w = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"), "voxcpm2");
        L.ffn_up_w = require(T, fn("ffn_up.weight"), "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"), "voxcpm2");
    }

    // LocDiT
    W.locdit_in_proj_w = require(T, "locdit.in_proj.weight", "voxcpm2");
    W.locdit_in_proj_b = require(T, "locdit.in_proj.bias", "voxcpm2");
    W.locdit_cond_proj_w = require(T, "locdit.cond_proj.weight", "voxcpm2");
    W.locdit_cond_proj_b = require(T, "locdit.cond_proj.bias", "voxcpm2");
    W.locdit_time_mlp_0_w = require(T, "locdit.time_mlp.0.weight", "voxcpm2");
    W.locdit_time_mlp_0_b = require(T, "locdit.time_mlp.0.bias", "voxcpm2");
    W.locdit_time_mlp_1_w = require(T, "locdit.time_mlp.1.weight", "voxcpm2");
    W.locdit_time_mlp_1_b = require(T, "locdit.time_mlp.1.bias", "voxcpm2");
    W.locdit_dt_mlp_0_w = require(T, "locdit.dt_mlp.0.weight", "voxcpm2");
    W.locdit_dt_mlp_0_b = require(T, "locdit.dt_mlp.0.bias", "voxcpm2");
    W.locdit_dt_mlp_1_w = require(T, "locdit.dt_mlp.1.weight", "voxcpm2");
    W.locdit_dt_mlp_1_b = require(T, "locdit.dt_mlp.1.bias", "voxcpm2");
    W.locdit_norm_w = try_get(T, "locdit.output_norm.weight");
    W.locdit_out_proj_w = require(T, "locdit.out_proj.weight", "voxcpm2");
    W.locdit_out_proj_b = require(T, "locdit.out_proj.bias", "voxcpm2");
    W.locdit_layers.resize(hp.locdit_n_layers);
    for (uint32_t i = 0; i < hp.locdit_n_layers; i++) {
        auto& L = W.locdit_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "locdit.blk.%u.%s", i, suffix);
            return nb;
        };
        L.norm1_w = require(T, fn("attn_norm.weight"), "voxcpm2");
        L.norm2_w = require(T, fn("ffn_norm.weight"), "voxcpm2");
        L.attn_q_w = require(T, fn("attn_q.weight"), "voxcpm2");
        L.attn_k_w = require(T, fn("attn_k.weight"), "voxcpm2");
        L.attn_v_w = require(T, fn("attn_v.weight"), "voxcpm2");
        L.attn_o_w = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"), "voxcpm2");
        L.ffn_up_w = require(T, fn("ffn_up.weight"), "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"), "voxcpm2");
    }

    // Projection heads
    W.enc_to_lm_w = try_get(T, "proj.enc_to_lm.weight");
    W.enc_to_lm_b = try_get(T, "proj.enc_to_lm.bias");
    W.lm_to_dit_w = try_get(T, "proj.lm_to_dit.weight");
    W.lm_to_dit_b = try_get(T, "proj.lm_to_dit.bias");
    W.res_to_dit_w = try_get(T, "proj.res_to_dit.weight");
    W.res_to_dit_b = try_get(T, "proj.res_to_dit.bias");
    W.fusion_w = try_get(T, "proj.fusion.weight");
    W.fusion_b = try_get(T, "proj.fusion.bias");

    // Stop predictor
    W.stop_proj_w = try_get(T, "stop.proj.weight");
    W.stop_proj_b = try_get(T, "stop.proj.bias");
    W.stop_head_w = try_get(T, "stop.head.weight"); // [2048, 2], no bias

    // VAE decoder (graceful degradation when absent)
    // vae_decode() accesses ctx->tensors directly; these fields are kept for reference.
    W.vae_in_conv.w = try_get(T, "vae.dec.layer.0.weight_v");
    W.vae_in_conv.b = try_get(T, "vae.dec.layer.0.bias");
    W.vae_out_conv.w = try_get(T, "vae.dec.layer.9.weight_v");
    W.vae_out_conv.b = try_get(T, "vae.dec.layer.9.bias");
    W.vae_out_snake_a = try_get(T, "vae.dec.layer.8.alpha");
    W.vae_sr_cond_w = try_get(T, "vae.dec.sr_cond.2.scale_embed");
    W.vae_sr_cond_b = try_get(T, "vae.dec.sr_cond.2.bias_embed");

    // KV caches
    const int kv_max_ctx = 4096;
    ctx->tslm_kv.init((int)hp.tslm_n_layers, (int)hp.tslm_n_kv, (int)hp.tslm_head_dim, kv_max_ctx);
    ctx->ralm_kv.init((int)hp.ralm_n_layers, (int)hp.ralm_n_kv, (int)hp.ralm_head_dim, kv_max_ctx);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: loaded — TSLM %uL d=%u n_kv=%u, RALM %uL, LocEnc %uL d=%u, LocDiT %uL\n",
                hp.tslm_n_layers, hp.tslm_d_model, hp.tslm_n_kv, hp.ralm_n_layers, hp.locenc_n_layers,
                hp.locenc_d_model, hp.locdit_n_layers);
        fprintf(stderr, "voxcpm2: vocab %zu tokens, audio_start=%u\n", ctx->tokenizer.id_to_token.size(),
                hp.audio_start_token);
    }

    // Set ggml matmul thread count
    g_cpu_n_threads = 4; // TODO: fix ctx->params

    return true;
}

// ---------------------------------------------------------------------------
// Shared prefill input builder — used by both `vox_synthesize_internal` and
// the cloning-aware diff stage handlers. Encapsulates `_make_ref_prefix` +
// per-position embed construction so the two call sites can't drift.
//
//   text         : synth text (UTF-8); will be tokenised + audio_start appended.
//   ref_samples  : 16 kHz mono float32 PCM, or nullptr for zero-shot.
//   ref_n_samples: length of `ref_samples` (samples, not bytes).
//
// Result fields are unset/empty on failure (e.g. empty text).
// ---------------------------------------------------------------------------
struct vox_prefill_inputs {
    std::vector<int32_t> all_tokens;
    std::vector<uint8_t> audio_mask_pos; // 1 = ref-audio position, 0 = text position
    std::vector<float> combined_embed;   // [N_pos * d_tslm]
    std::vector<float> feat_embed_pos;   // [N_pos * d_tslm]; nonzero only at audio positions
    std::vector<float> ref_feat;         // [T_ref * P * D] from VAE encoder (kept for AR-loop access)
    int T_ref = 0;
    int N_pos = 0;
    bool have_ref = false;
};

// Process-wide cache so that the diff harness (which calls extract_stage
// once per stage) doesn't recompute the 69-patch LocEnc / 82-position TSLM
// prefill 6× per invocation. Keyed on (ctx, text, ref_n_samples) — that's
// enough to distinguish reasonable use; same-text/same-ref repeated calls
// reuse the cached inputs. The cache lifetime is the process — single-call
// CLI use never hits a second lookup.
struct vox_prefill_cache_key {
    voxcpm2_context* ctx;
    std::string text;
    int ref_n_samples;
    bool operator==(const vox_prefill_cache_key& o) const {
        return ctx == o.ctx && ref_n_samples == o.ref_n_samples && text == o.text;
    }
};

static vox_prefill_inputs build_prefill_inputs_impl(voxcpm2_context* ctx, const std::string& text,
                                                    const float* ref_samples, int ref_n_samples, ggml_backend_t cpu_be);

static const vox_prefill_inputs& build_prefill_inputs_cached(voxcpm2_context* ctx, const std::string& text,
                                                             const float* ref_samples, int ref_n_samples,
                                                             ggml_backend_t cpu_be) {
    static vox_prefill_cache_key cached_key{nullptr, std::string(), -1};
    static vox_prefill_inputs cached;
    vox_prefill_cache_key key{ctx, text, ref_n_samples};
    if (!(key == cached_key) || cached.N_pos == 0) {
        cached = build_prefill_inputs_impl(ctx, text, ref_samples, ref_n_samples, cpu_be);
        cached_key = key;
    }
    return cached;
}

static vox_prefill_inputs build_prefill_inputs(voxcpm2_context* ctx, const std::string& text, const float* ref_samples,
                                               int ref_n_samples, ggml_backend_t cpu_be) {
    return build_prefill_inputs_impl(ctx, text, ref_samples, ref_n_samples, cpu_be);
}

static vox_prefill_inputs build_prefill_inputs_impl(voxcpm2_context* ctx, const std::string& text,
                                                    const float* ref_samples, int ref_n_samples,
                                                    ggml_backend_t cpu_be) {
    vox_prefill_inputs out;
    const auto& hp = ctx->hp;
    int d_tslm = (int)hp.tslm_d_model;
    int d_dit = (int)hp.locdit_d_model;
    int P_frames = (int)hp.patch_frames;
    int feat_dim_vae = 64;
    const bool use_graph = vox_env_bool_default_on("VOXCPM2_USE_GRAPH");

    // 1. Tokenise (vox_tokenize already appends audio_start_token).
    std::vector<int32_t> text_tokens = vox_tokenize(ctx->tokenizer, text);
    if (text_tokens.empty()) {
        return out;
    }

    // 2. VAE-encode the reference WAV if provided.
    out.have_ref =
        (ref_samples != nullptr && ref_n_samples > 0 && hp.ref_audio_start_token != 0 && hp.ref_audio_end_token != 0);
    if (out.have_ref) {
        out.ref_feat = vae_encode(ctx, ref_samples, ref_n_samples, &out.T_ref);
        if (out.T_ref <= 0) {
            fprintf(stderr, "voxcpm2: VAE encoder produced 0 patches — falling back to zero-shot\n");
            out.have_ref = false;
            out.ref_feat.clear();
        }
    }

    // 3. Build `_make_ref_prefix` + concat with text tokens.
    if (out.have_ref) {
        out.all_tokens.reserve((size_t)out.T_ref + 2 + text_tokens.size());
        out.audio_mask_pos.reserve((size_t)out.T_ref + 2 + text_tokens.size());
        out.all_tokens.push_back((int32_t)hp.ref_audio_start_token);
        out.audio_mask_pos.push_back(0);
        for (int i = 0; i < out.T_ref; i++) {
            out.all_tokens.push_back(0);
            out.audio_mask_pos.push_back(1);
        }
        out.all_tokens.push_back((int32_t)hp.ref_audio_end_token);
        out.audio_mask_pos.push_back(0);
        for (int32_t tok : text_tokens) {
            out.all_tokens.push_back(tok);
            out.audio_mask_pos.push_back(0);
        }
    } else {
        out.all_tokens = text_tokens;
        out.audio_mask_pos.assign(text_tokens.size(), 0);
    }
    out.N_pos = (int)out.all_tokens.size();

    // 4. Per-position embed: combined_embed[t] = text_mask*embed_tokens + audio_mask*enc_to_lm(locenc(audio_feat[t])).
    out.combined_embed.assign((size_t)out.N_pos * d_tslm, 0.0f);
    out.feat_embed_pos.assign((size_t)out.N_pos * d_tslm, 0.0f);
    for (int t = 0; t < out.N_pos; t++) {
        if (out.audio_mask_pos[t]) {
            int patch_idx = t - 1;
            const float* patch = out.ref_feat.data() + (size_t)patch_idx * P_frames * feat_dim_vae;
            std::vector<float> enc_out =
                use_graph ? locenc_forward_graph(ctx, patch) : locenc_forward(ctx, patch, cpu_be);
            if (ctx->weights.enc_to_lm_w && ctx->weights.enc_to_lm_b) {
                matmul_mv_bias(cpu_be, ctx->weights.enc_to_lm_w, ctx->weights.enc_to_lm_b, enc_out.data(), d_dit,
                               out.feat_embed_pos.data() + (size_t)t * d_tslm, d_tslm);
            } else {
                int copy_d = std::min(d_dit, d_tslm);
                std::memcpy(out.feat_embed_pos.data() + (size_t)t * d_tslm, enc_out.data(),
                            (size_t)copy_d * sizeof(float));
            }
            std::memcpy(out.combined_embed.data() + (size_t)t * d_tslm, out.feat_embed_pos.data() + (size_t)t * d_tslm,
                        (size_t)d_tslm * sizeof(float));
        } else {
            int id = out.all_tokens[t];
            if (id < 0 || id >= (int)hp.n_vocab) {
                id = 0;
            }
            get_row_f32(ctx->weights.tslm_token_embd, id, out.combined_embed.data() + (size_t)t * d_tslm);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Core synthesis pipeline
// ---------------------------------------------------------------------------

static float* vox_synthesize_internal(voxcpm2_context* ctx, const char* text, const float* ref_samples,
                                      int ref_n_samples, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    ggml_backend_t cpu_be = get_cpu_backend();
    if (ggml_backend_is_cpu(cpu_be)) {
        ggml_backend_cpu_set_n_threads(cpu_be, ctx->n_threads);
    }

    // Seed RNG for CFM noise
    mt19937_seed(ctx->rng, ctx->seed != 0 ? ctx->seed : 42);

    double t0_total = vox_now_ms();
    const auto& hp = ctx->hp;
    int d_tslm = (int)hp.tslm_d_model;
    int d_dit = (int)hp.locdit_d_model;
    int P_frames = (int)hp.patch_frames;
    int feat_dim_vae = 64;

    // 1. Build prefill inputs (tokens + masks + combined embeds + ref feats).
    vox_prefill_inputs pi = build_prefill_inputs(ctx, std::string(text), ref_samples, ref_n_samples, cpu_be);
    if (pi.N_pos == 0) {
        fprintf(stderr, "voxcpm2: empty token sequence\n");
        return nullptr;
    }
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: tokenized '%s' -> %d positions%s\n", text, pi.N_pos,
                pi.have_ref ? " (incl ref)" : "");
    }
    int N_pos = pi.N_pos;
    bool have_ref = pi.have_ref;
    const auto& audio_mask_pos = pi.audio_mask_pos;
    const auto& feat_embed_pos = pi.feat_embed_pos;

    // 2. TSLM prefill from the combined embeds (capture all positions for RALM).
    double t0_prefill = vox_now_ms();
    std::vector<float> all_pos;
    tslm_prefill_hooks hooks;
    hooks.max_capture_positions = N_pos;
    hooks.all_positions = &all_pos;
    tslm_prefill_from_embeds(ctx, pi.combined_embed.data(), N_pos, cpu_be, hooks);

    // 5. Apply TSLM output norm per position.
    std::vector<float> normed_all((size_t)N_pos * d_tslm);
    for (int i = 0; i < N_pos; i++) {
        rms_norm_cpu(all_pos.data() + (size_t)i * d_tslm, tensor_data_f32(ctx->weights.tslm_output_norm),
                     normed_all.data() + (size_t)i * d_tslm, d_tslm, hp.rms_norm_eps);
    }
    // 5b. FSQ masking — Python:
    //   enc_outputs = fsq(enc_outputs) * audio_mask + enc_outputs * text_mask
    // For audio positions (ref patches), replace with FSQ-quantised version.
    if (have_ref) {
        for (int t = 0; t < N_pos; t++) {
            if (audio_mask_pos[t]) {
                std::vector<float> fsq_pos = fsq_forward(ctx, normed_all.data() + (size_t)t * d_tslm, cpu_be);
                std::memcpy(normed_all.data() + (size_t)t * d_tslm, fsq_pos.data(), (size_t)d_tslm * sizeof(float));
            }
        }
    }
    std::vector<float> tslm_hidden(normed_all.data() + (size_t)(N_pos - 1) * d_tslm,
                                   normed_all.data() + (size_t)N_pos * d_tslm);
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: TSLM prefill %.1f ms (%d positions%s)\n", vox_now_ms() - t0_prefill, N_pos,
                have_ref ? " incl. ref" : "");
    }

    // 6. fusion_concat_proj + multi-position RALM prefill. Python concatenates
    //    `enc_outputs` with `audio_mask * feat_embed` along the channel axis,
    //    so text positions get a zero second half and ref positions get the
    //    enc_to_lm projection of their LocEnc output.
    std::vector<float> ralm_hidden;
    {
        int in_dim = 2 * d_tslm;
        std::vector<float> ralm_input((size_t)N_pos * d_tslm);
        for (int i = 0; i < N_pos; i++) {
            std::vector<float> cat_buf(in_dim, 0.0f);
            std::memcpy(cat_buf.data(), normed_all.data() + (size_t)i * d_tslm, (size_t)d_tslm * sizeof(float));
            if (audio_mask_pos[i]) {
                std::memcpy(cat_buf.data() + d_tslm, feat_embed_pos.data() + (size_t)i * d_tslm,
                            (size_t)d_tslm * sizeof(float));
            }
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, cat_buf.data(), in_dim,
                           ralm_input.data() + (size_t)i * d_tslm, d_tslm);
        }
        std::vector<float> ralm_out = ralm_prefill_multi(ctx, ralm_input.data(), N_pos, cpu_be);
        int dr = (int)hp.ralm_d_model;
        ralm_hidden.resize(dr);
        rms_norm_cpu(ralm_out.data() + (size_t)(N_pos - 1) * dr, tensor_data_f32(ctx->weights.ralm_output_norm),
                     ralm_hidden.data(), dr, hp.rms_norm_eps);
    }

    // 5. Build mu [2048] for LocDiT: mu = cat(lm_to_dit(tslm), res_to_dit(ralm))
    // lm_to_dit:  [2048 → 1024] maps TSLM hidden to first half of mu
    // res_to_dit: [2048 → 1024] maps RALM hidden to second half of mu
    // mu.view(-1, 1024) = [2, 1024] = 2 conditioning tokens in the DiT sequence
    int d_lm = d_tslm;                 // 2048 (alias kept for readability below)
    int d_ralm = (int)hp.ralm_d_model; // 2048
    int d_mu = 2 * d_dit;              // 2048 = full mu

    auto build_mu = [&](const std::vector<float>& lm_h, const std::vector<float>& ralm_h) -> std::vector<float> {
        std::vector<float> mu(d_mu, 0.0f);
        // First half: lm_to_dit projection [d_lm → d_dit]
        if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b, lm_h.data(), d_lm, mu.data(),
                           d_dit);
        }
        // Second half: res_to_dit projection [d_ralm → d_dit]
        if (ctx->weights.res_to_dit_w && ctx->weights.res_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.res_to_dit_w, ctx->weights.res_to_dit_b, ralm_h.data(), d_ralm,
                           mu.data() + d_dit, d_dit);
        }
        return mu;
    };

    std::vector<float> mu = build_mu(tslm_hidden, ralm_hidden);
    // cond_raw for LocDiT: Python computes `prefix_feat_cond = audio_feat[:, -1, ...]`
    // — the audio_feat at the LAST input position. The last input is always the
    // text's audio_start_token (text position with zero audio_feat) for both
    // zero-shot and ref-prefix synthesis, so this stays all-zeros either way.
    std::vector<float> prev_patch_raw((size_t)feat_dim_vae * P_frames, 0.0f);

    // FSQ output for AR loop (declared here so it's in scope)
    std::vector<float> fsq_out;

    // 6. AR loop
    double t0_ar = vox_now_ms();
    std::vector<std::vector<float>> patches;
    float stop_thresh = 0.5f;
    int step = 0;

    // Per-substep accumulators gated on VOXCPM2_BENCH=1. Cheap (one
    // vox_now_ms / step) but skips the prints when not requested.
    const bool bench = vox_env_bool("VOXCPM2_BENCH");
    double sum_cfm = 0, sum_locenc = 0, sum_enc_to_lm = 0;
    double sum_tslm = 0, sum_fsq = 0, sum_fusion = 0, sum_ralm = 0, sum_stop = 0;

    // VOXCPM2_USE_GRAPH=1: replace the 28-call tslm_layer_step loop with one
    // backend cgraph per AR step. Backend KV is one-time-synced from the
    // legacy vox_kv_cache (built by the legacy prefill path above) on first
    // use of the graph; subsequent steps write/read the backend KV directly
    // through the graph (no further CPU↔backend traffic). Resetting
    // tslm_kv_synced here ensures every synthesis call re-syncs from the
    // fresh prefill cache.
    const bool use_graph_tslm = vox_env_bool_default_on("VOXCPM2_USE_GRAPH");
    ctx->tslm_kv_synced = false;

    // Python AR loop order (from voxcpm2.py _inference, lines 1060-1108):
    //   1. Build mu → CFM solve → LocEnc → enc_to_lm → collect patch
    //   2. Stop check (on PREVIOUS lm_hidden, BEFORE TSLM step)
    //   3. TSLM step → FSQ → Fusion → RALM step → update lm_hidden/mu
    // Python min_len=2 by default (stop only if step > min_len).
    int min_len = 2;

    while (step < ctx->max_len) {
        double t0_step = vox_now_ms();

        // 1a. CFM Euler solve (LocDiT)
        double tb = bench ? vox_now_ms() : 0;
        std::vector<float> noise(feat_dim_vae * P_frames);
        fill_gaussian_noise_bf16(noise.data(), (int)noise.size(), ctx->rng);
        std::vector<float> patch = cfm_euler_solve(ctx, mu.data(), prev_patch_raw.data(), ctx->inference_steps,
                                                   ctx->cfg_value, cpu_be, noise.data());
        if (bench)
            sum_cfm += vox_now_ms() - tb;

        // 1b. Transpose patch [C=64, T=4] → [T=4, C=64]
        std::vector<float> patch_tf(feat_dim_vae * P_frames);
        for (int t = 0; t < P_frames; t++)
            for (int c = 0; c < feat_dim_vae; c++)
                patch_tf[t * feat_dim_vae + c] = patch[c * P_frames + t];

        // 1c. LocEnc on predicted patch
        tb = bench ? vox_now_ms() : 0;
        std::vector<float> enc_out =
            use_graph_tslm ? locenc_forward_graph(ctx, patch_tf.data()) : locenc_forward(ctx, patch_tf.data(), cpu_be);
        if (bench)
            sum_locenc += vox_now_ms() - tb;

        // 1d. enc_to_lm projection
        tb = bench ? vox_now_ms() : 0;
        std::vector<float> enc_lm(d_lm, 0.0f);
        if (ctx->weights.enc_to_lm_w && ctx->weights.enc_to_lm_b) {
            matmul_mv_bias(cpu_be, ctx->weights.enc_to_lm_w, ctx->weights.enc_to_lm_b, enc_out.data(), d_dit,
                           enc_lm.data(), d_lm);
        } else {
            int copy_d = std::min(d_dit, d_lm);
            std::memcpy(enc_lm.data(), enc_out.data(), (size_t)copy_d * sizeof(float));
        }
        if (bench)
            sum_enc_to_lm += vox_now_ms() - tb;

        // 1e. Collect patch + update cond for next step
        patches.push_back(patch_tf);
        prev_patch_raw = patch_tf;

        // 2. Stop check — BEFORE TSLM step, using PREVIOUS tslm_hidden
        // Python: stop_head(stop_actn(stop_proj(lm_hidden))).argmax() == 1
        // At step 0, tslm_hidden is the normed prefill output (not FSQ'd).
        // At step >0, tslm_hidden is the FSQ'd output from the previous step.
        {
            tb = bench ? vox_now_ms() : 0;
            float sp = stop_score(ctx, tslm_hidden.data(), cpu_be);
            if (bench)
                sum_stop += vox_now_ms() - tb;
            if (ctx->verbosity >= 2) {
                fprintf(stderr, "voxcpm2: step %d stop=%.3f (%.1f ms)\n", step, sp, vox_now_ms() - t0_step);
            }
            if (sp > stop_thresh && step > min_len) {
                if (ctx->verbosity >= 1) {
                    fprintf(stderr, "voxcpm2: stopped at step %d (stop=%.3f)\n", step + 1, sp);
                }
                break;
            }
        }

        // 3a. TSLM step (single audio token position)
        tb = bench ? vox_now_ms() : 0;
        {
            int tslm_pos = ctx->tslm_kv.n_past;
            if (use_graph_tslm) {
                if (!ctx->tslm_kv_synced) {
                    if (!init_tslm_kv_backend(ctx)) {
                        fprintf(stderr, "voxcpm2: tslm kv backend init failed; falling back to legacy step\n");
                    } else {
                        sync_tslm_kv_cpu_to_backend(ctx);
                        ctx->tslm_kv_synced = true;
                    }
                }
                tslm_hidden = tslm_step_graph(ctx, enc_lm.data(), tslm_pos);
            } else {
                std::vector<float> h = enc_lm;
                for (int l = 0; l < (int)ctx->hp.tslm_n_layers; l++) {
                    tslm_layer_step(ctx, l, h.data(), tslm_pos, cpu_be);
                }
                std::vector<float> normed(d_lm);
                rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d_lm,
                             ctx->hp.rms_norm_eps);
                tslm_hidden = normed;
            }
            ctx->tslm_kv.n_past++;
        }
        if (bench)
            sum_tslm += vox_now_ms() - tb;

        // 3b. FSQ
        tb = bench ? vox_now_ms() : 0;
        fsq_out = fsq_forward(ctx, tslm_hidden.data(), cpu_be);
        if (bench)
            sum_fsq += vox_now_ms() - tb;

        // 3c. Fusion: cat(fsq_out, enc_lm) → fusion_proj
        tb = bench ? vox_now_ms() : 0;
        std::vector<float> fusion_in((size_t)(d_lm + d_lm));
        std::memcpy(fusion_in.data(), fsq_out.data(), (size_t)d_lm * sizeof(float));
        std::memcpy(fusion_in.data() + d_lm, enc_lm.data(), (size_t)d_lm * sizeof(float));

        std::vector<float> fusion_out(d_ralm, 0.0f);
        if (ctx->weights.fusion_w && ctx->weights.fusion_b) {
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, fusion_in.data(), 2 * d_lm,
                           fusion_out.data(), d_ralm);
        } else {
            fusion_out = fsq_out;
        }
        if (bench)
            sum_fusion += vox_now_ms() - tb;

        // 3d. RALM step
        tb = bench ? vox_now_ms() : 0;
        {
            std::vector<float> h = fusion_out;
            for (int l = 0; l < (int)ctx->hp.ralm_n_layers; l++) {
                ralm_layer_step(ctx, l, h.data(), cpu_be);
            }
            ctx->ralm_kv.n_past++;

            std::vector<float> normed(d_ralm);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.ralm_output_norm), normed.data(), d_ralm,
                         ctx->hp.rms_norm_eps);
            ralm_hidden = normed;
        }
        if (bench)
            sum_ralm += vox_now_ms() - tb;

        // 3e. Update: tslm_hidden = FSQ'd for next step's mu + stop check
        tslm_hidden = fsq_out;
        mu = build_mu(tslm_hidden, ralm_hidden);
        step++;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: AR loop %d steps, %.1f ms\n", step, vox_now_ms() - t0_ar);
    }
    if (bench && step > 0) {
        double tot = sum_cfm + sum_locenc + sum_enc_to_lm + sum_tslm + sum_fsq + sum_fusion + sum_ralm + sum_stop;
        fprintf(stderr, "voxcpm2[bench]: AR per-step avg (over %d steps, %.1f ms total):\n", step,
                vox_now_ms() - t0_ar);
        fprintf(stderr, "voxcpm2[bench]:   cfm        %7.1f ms  (%5.1f%%)\n", sum_cfm / step, 100.0 * sum_cfm / tot);
        fprintf(stderr, "voxcpm2[bench]:   locenc     %7.1f ms  (%5.1f%%)\n", sum_locenc / step,
                100.0 * sum_locenc / tot);
        fprintf(stderr, "voxcpm2[bench]:   enc_to_lm  %7.1f ms  (%5.1f%%)\n", sum_enc_to_lm / step,
                100.0 * sum_enc_to_lm / tot);
        fprintf(stderr, "voxcpm2[bench]:   tslm_step  %7.1f ms  (%5.1f%%)\n", sum_tslm / step, 100.0 * sum_tslm / tot);
        fprintf(stderr, "voxcpm2[bench]:   fsq        %7.1f ms  (%5.1f%%)\n", sum_fsq / step, 100.0 * sum_fsq / tot);
        fprintf(stderr, "voxcpm2[bench]:   fusion     %7.1f ms  (%5.1f%%)\n", sum_fusion / step,
                100.0 * sum_fusion / tot);
        fprintf(stderr, "voxcpm2[bench]:   ralm_step  %7.1f ms  (%5.1f%%)\n", sum_ralm / step, 100.0 * sum_ralm / tot);
        fprintf(stderr, "voxcpm2[bench]:   stop_pred  %7.1f ms  (%5.1f%%)\n", sum_stop / step, 100.0 * sum_stop / tot);
    }

    // 7. VAE decode
    double t0_vae = vox_now_ms();
    std::vector<float> pcm = vae_decode(ctx, patches, cpu_be);
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: VAE decode %.1f ms -> %zu samples @48kHz\n", vox_now_ms() - t0_vae, pcm.size());
        fprintf(stderr, "voxcpm2: total %.1f ms\n", vox_now_ms() - t0_total);
    }

    if (pcm.empty()) {
        fprintf(stderr, "voxcpm2: empty audio output\n");
        return nullptr;
    }

    *out_n_samples = (int)pcm.size();
    float* result = (float*)std::malloc(pcm.size() * sizeof(float));
    if (!result)
        return nullptr;
    std::memcpy(result, pcm.data(), pcm.size() * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

struct voxcpm2_context_params voxcpm2_context_default_params(void) {
    struct voxcpm2_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.flash_attn = true;
    p.inference_steps = 10;
    p.cfg_value = 2.0f;
    p.max_len = 2000;
    p.seed = 0;
    return p;
}

struct voxcpm2_context* voxcpm2_init_from_file(const char* path_model, struct voxcpm2_context_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new voxcpm2_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->use_gpu = params.use_gpu;
    ctx->flash_attn = params.flash_attn;
    ctx->inference_steps = params.inference_steps > 0 ? params.inference_steps : 10;
    ctx->cfg_value = params.cfg_value > 0.0f ? params.cfg_value : 2.0f;
    ctx->max_len = params.max_len > 0 ? params.max_len : 2000;
    ctx->seed = params.seed;

    // Backend pool. With `use_gpu`, init_best picks Metal / Vulkan / CUDA.
    // On Apple Silicon, Metal allocates in unified-memory "shared" mode, so
    // `tensor->data` stays CPU-readable and the remaining legacy CPU paths
    // (matmul_mv_ggml, rms_norm_cpu, …) work against the same pointers the
    // graph paths use. On discrete GPUs (Vulkan, CUDA) the default buffer
    // is device-local VRAM — CPU can't dereference tensor->data. Instead
    // we load weights to CPU for legacy paths and create GPU mirror copies
    // for graph-build functions, giving both worlds native-speed access
    // with no cross-backend copies at compute time.
    ctx->backend_cpu = get_cpu_backend();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "voxcpm2: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    if (params.use_gpu) {
        ctx->backend = ggml_backend_init_best();
        if (!ctx->backend) {
            if (params.verbosity >= 1) {
                fprintf(stderr, "voxcpm2: best backend unavailable, falling back to CPU\n");
            }
            ctx->backend = ctx->backend_cpu;
        }
    } else {
        ctx->backend = ctx->backend_cpu;
    }

    // Detect whether the GPU backend's buffer is host-visible. If not,
    // weights will be loaded to CPU and mirrored to GPU (see below).
    bool needs_gpu_mirror = false;
    if (ctx->backend != ctx->backend_cpu) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx->backend);
        needs_gpu_mirror = (buft && !ggml_backend_buft_is_host(buft));
    }

    if (!vox_load_weights(ctx, path_model)) {
        fprintf(stderr, "voxcpm2: failed to load '%s'\n", path_model);
        voxcpm2_free(ctx);
        return nullptr;
    }

    // On discrete GPUs: weights were loaded to CPU (vox_load_weights
    // checks ggml_backend_buft_is_host internally). Create GPU copies
    // for the graph-build paths so ggml_backend_graph_compute runs
    // entirely on the GPU with zero cross-backend data movement.
    if (needs_gpu_mirror) {
        ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(ctx->backend);

        // 1. Allocate a GPU ggml context with mirror tensors.
        size_t ctx_size = ctx->tensors.size() * ggml_tensor_overhead() + 1024;
        ggml_init_params ip = {ctx_size, nullptr, /*no_alloc=*/true};
        ctx->gpu_ggml_ctx = ggml_init(ip);
        if (!ctx->gpu_ggml_ctx) {
            fprintf(stderr, "voxcpm2: failed to create GPU mirror context\n");
            voxcpm2_free(ctx);
            return nullptr;
        }

        for (auto& [name, cpu_t] : ctx->tensors) {
            ggml_tensor* gpu_t = ggml_dup_tensor(ctx->gpu_ggml_ctx, cpu_t);
            ggml_set_name(gpu_t, name.c_str());
            ctx->gpu_tensors[name] = gpu_t;
        }

        // 2. Allocate GPU buffer for all mirror tensors.
        ctx->gpu_weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx->gpu_ggml_ctx, gpu_buft);
        if (!ctx->gpu_weight_buf) {
            fprintf(stderr, "voxcpm2: failed to allocate GPU weight buffer — "
                            "falling back to CPU\n");
            ggml_free(ctx->gpu_ggml_ctx);
            ctx->gpu_ggml_ctx = nullptr;
            ctx->gpu_tensors.clear();
            ctx->backend = ctx->backend_cpu;
        } else {
            // 3. Copy data CPU → GPU.
            for (auto& [name, cpu_t] : ctx->tensors) {
                auto it = ctx->gpu_tensors.find(name);
                if (it != ctx->gpu_tensors.end()) {
                    ggml_backend_tensor_copy(cpu_t, it->second);
                }
            }

            // 4. Populate gpu_weights struct by looking up GPU tensors.
            vox_weights& GW = ctx->gpu_weights;
            auto& GT = ctx->gpu_tensors;
            auto gt = [&](const char* n) -> ggml_tensor* {
                auto it = GT.find(n);
                return it != GT.end() ? it->second : nullptr;
            };
            GW.tslm_token_embd = gt("tslm.token_embd.weight");
            GW.tslm_output_norm = gt("tslm.output_norm.weight");
            GW.tslm_rope_short = gt("tslm.rope_short_factors");
            GW.tslm_rope_long = gt("tslm.rope_long_factors");
            GW.tslm_layers.resize(ctx->hp.tslm_n_layers);
            for (uint32_t i = 0; i < ctx->hp.tslm_n_layers; i++) {
                auto& L = GW.tslm_layers[i];
                char nb[256];
                auto fn = [&](const char* sfx) { snprintf(nb, sizeof(nb), "tslm.blk.%u.%s", i, sfx); return nb; };
                L.attn_norm_w = gt(fn("attn_norm.weight"));
                L.attn_q_w = gt(fn("attn_q.weight"));
                L.attn_k_w = gt(fn("attn_k.weight"));
                L.attn_v_w = gt(fn("attn_v.weight"));
                L.attn_o_w = gt(fn("attn_output.weight"));
                L.ffn_norm_w = gt(fn("ffn_norm.weight"));
                L.ffn_gate_w = gt(fn("ffn_gate.weight"));
                L.ffn_up_w = gt(fn("ffn_up.weight"));
                L.ffn_down_w = gt(fn("ffn_down.weight"));
            }
            GW.ralm_output_norm = gt("ralm.output_norm.weight");
            GW.ralm_layers.resize(ctx->hp.ralm_n_layers);
            for (uint32_t i = 0; i < ctx->hp.ralm_n_layers; i++) {
                auto& L = GW.ralm_layers[i];
                char nb[256];
                auto fn = [&](const char* sfx) { snprintf(nb, sizeof(nb), "ralm.blk.%u.%s", i, sfx); return nb; };
                L.attn_norm_w = gt(fn("attn_norm.weight"));
                L.attn_q_w = gt(fn("attn_q.weight"));
                L.attn_k_w = gt(fn("attn_k.weight"));
                L.attn_v_w = gt(fn("attn_v.weight"));
                L.attn_o_w = gt(fn("attn_output.weight"));
                L.ffn_norm_w = gt(fn("ffn_norm.weight"));
                L.ffn_gate_w = gt(fn("ffn_gate.weight"));
                L.ffn_up_w = gt(fn("ffn_up.weight"));
                L.ffn_down_w = gt(fn("ffn_down.weight"));
            }
            GW.fsq_in_proj_w = gt("fsq.in_proj.weight");
            GW.fsq_in_proj_b = gt("fsq.in_proj.bias");
            GW.fsq_out_proj_w = gt("fsq.out_proj.weight");
            GW.fsq_out_proj_b = gt("fsq.out_proj.bias");
            GW.locenc_cls_token = gt("locenc.cls_token");
            GW.locenc_in_proj_w = gt("locenc.in_proj.weight");
            GW.locenc_in_proj_b = gt("locenc.in_proj.bias");
            GW.locenc_norm_w = gt("locenc.output_norm.weight");
            GW.locenc_layers.resize(ctx->hp.locenc_n_layers);
            for (uint32_t i = 0; i < ctx->hp.locenc_n_layers; i++) {
                auto& L = GW.locenc_layers[i];
                char nb[256];
                auto fn = [&](const char* sfx) { snprintf(nb, sizeof(nb), "locenc.blk.%u.%s", i, sfx); return nb; };
                L.norm1_w = gt(fn("attn_norm.weight"));
                L.norm2_w = gt(fn("ffn_norm.weight"));
                L.attn_q_w = gt(fn("attn_q.weight"));
                L.attn_k_w = gt(fn("attn_k.weight"));
                L.attn_v_w = gt(fn("attn_v.weight"));
                L.attn_o_w = gt(fn("attn_output.weight"));
                L.ffn_gate_w = gt(fn("ffn_gate.weight"));
                L.ffn_up_w = gt(fn("ffn_up.weight"));
                L.ffn_down_w = gt(fn("ffn_down.weight"));
            }
            GW.locdit_in_proj_w = gt("locdit.in_proj.weight");
            GW.locdit_in_proj_b = gt("locdit.in_proj.bias");
            GW.locdit_cond_proj_w = gt("locdit.cond_proj.weight");
            GW.locdit_cond_proj_b = gt("locdit.cond_proj.bias");
            GW.locdit_time_mlp_0_w = gt("locdit.time_mlp.0.weight");
            GW.locdit_time_mlp_0_b = gt("locdit.time_mlp.0.bias");
            GW.locdit_time_mlp_1_w = gt("locdit.time_mlp.1.weight");
            GW.locdit_time_mlp_1_b = gt("locdit.time_mlp.1.bias");
            GW.locdit_dt_mlp_0_w = gt("locdit.dt_mlp.0.weight");
            GW.locdit_dt_mlp_0_b = gt("locdit.dt_mlp.0.bias");
            GW.locdit_dt_mlp_1_w = gt("locdit.dt_mlp.1.weight");
            GW.locdit_dt_mlp_1_b = gt("locdit.dt_mlp.1.bias");
            GW.locdit_norm_w = gt("locdit.output_norm.weight");
            GW.locdit_out_proj_w = gt("locdit.out_proj.weight");
            GW.locdit_out_proj_b = gt("locdit.out_proj.bias");
            GW.locdit_layers.resize(ctx->hp.locdit_n_layers);
            for (uint32_t i = 0; i < ctx->hp.locdit_n_layers; i++) {
                auto& L = GW.locdit_layers[i];
                char nb[256];
                auto fn = [&](const char* sfx) { snprintf(nb, sizeof(nb), "locdit.blk.%u.%s", i, sfx); return nb; };
                L.norm1_w = gt(fn("attn_norm.weight"));
                L.norm2_w = gt(fn("ffn_norm.weight"));
                L.attn_q_w = gt(fn("attn_q.weight"));
                L.attn_k_w = gt(fn("attn_k.weight"));
                L.attn_v_w = gt(fn("attn_v.weight"));
                L.attn_o_w = gt(fn("attn_output.weight"));
                L.ffn_gate_w = gt(fn("ffn_gate.weight"));
                L.ffn_up_w = gt(fn("ffn_up.weight"));
                L.ffn_down_w = gt(fn("ffn_down.weight"));
            }
            GW.enc_to_lm_w = gt("proj.enc_to_lm.weight");
            GW.enc_to_lm_b = gt("proj.enc_to_lm.bias");
            GW.lm_to_dit_w = gt("proj.lm_to_dit.weight");
            GW.lm_to_dit_b = gt("proj.lm_to_dit.bias");
            GW.res_to_dit_w = gt("proj.res_to_dit.weight");
            GW.res_to_dit_b = gt("proj.res_to_dit.bias");
            GW.fusion_w = gt("proj.fusion.weight");
            GW.fusion_b = gt("proj.fusion.bias");
            GW.stop_proj_w = gt("stop.proj.weight");
            GW.stop_proj_b = gt("stop.proj.bias");
            GW.stop_head_w = gt("stop.head.weight");

            ctx->has_gpu_weights = true;
            if (params.verbosity >= 1) {
                fprintf(stderr, "voxcpm2: created GPU weight mirror on %s (%.0f MB)\n",
                        ggml_backend_name(ctx->backend),
                        (double)ggml_backend_buffer_get_size(ctx->gpu_weight_buf) / (1024.0 * 1024.0));
            }
        }
    }

    // Generous arena: ~9 nodes/layer × 12 LocDiT layers + I/O ≈ 130 nodes;
    // 28 TSLM layers ≈ 280 nodes. 4096 nodes leaves headroom for the
    // largest graph (TSLM 28L step) without re-allocing per call.
    const int max_nodes = 4096;
    ctx->compute_meta.resize(ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false));
    ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->galloc) {
        fprintf(stderr, "voxcpm2: failed to create gallocr\n");
        voxcpm2_free(ctx);
        return nullptr;
    }

    if (params.verbosity >= 1) {
        const char* be_name = ggml_backend_name(ctx->backend);
        fprintf(stderr, "voxcpm2: backend = %s\n", be_name ? be_name : "(null)");
    }

    return ctx;
}

void voxcpm2_free(struct voxcpm2_context* ctx) {
    if (!ctx)
        return;
    for (auto& bk : ctx->tslm_buckets) {
        if (bk.galloc) {
            ggml_gallocr_free(bk.galloc);
            bk.galloc = nullptr;
        }
        if (bk.arena_ctx) {
            ggml_free(bk.arena_ctx);
            bk.arena_ctx = nullptr;
        }
        bk.gf = nullptr;
        bk.arena_meta.clear();
    }
    if (ctx->locdit_galloc) {
        ggml_gallocr_free(ctx->locdit_galloc);
        ctx->locdit_galloc = nullptr;
    }
    if (ctx->locdit_arena_ctx) {
        ggml_free(ctx->locdit_arena_ctx);
        ctx->locdit_arena_ctx = nullptr;
    }
    if (ctx->locenc_galloc) {
        ggml_gallocr_free(ctx->locenc_galloc);
        ctx->locenc_galloc = nullptr;
    }
    if (ctx->locenc_arena_ctx) {
        ggml_free(ctx->locenc_arena_ctx);
        ctx->locenc_arena_ctx = nullptr;
    }
    if (ctx->tslm_kv_buf) {
        ggml_backend_buffer_free(ctx->tslm_kv_buf);
        ctx->tslm_kv_buf = nullptr;
    }
    if (ctx->tslm_kv_ctx) {
        ggml_free(ctx->tslm_kv_ctx);
        ctx->tslm_kv_ctx = nullptr;
    }
    if (ctx->vae_wn_ggml_buf) {
        ggml_backend_buffer_free(ctx->vae_wn_ggml_buf);
        ctx->vae_wn_ggml_buf = nullptr;
    }
    if (ctx->vae_wn_ggml_ctx) {
        ggml_free(ctx->vae_wn_ggml_ctx);
        ctx->vae_wn_ggml_ctx = nullptr;
    }
    ctx->vae_wn_ggml_tensors.clear();
    if (ctx->galloc) {
        ggml_gallocr_free(ctx->galloc);
        ctx->galloc = nullptr;
    }
    // GPU weight mirrors (discrete GPU only)
    ctx->gpu_tensors.clear();
    if (ctx->gpu_weight_buf) {
        ggml_backend_buffer_free(ctx->gpu_weight_buf);
        ctx->gpu_weight_buf = nullptr;
    }
    if (ctx->gpu_ggml_ctx) {
        ggml_free(ctx->gpu_ggml_ctx);
        ctx->gpu_ggml_ctx = nullptr;
    }
    if (ctx->weight_buf) {
        ggml_backend_buffer_free(ctx->weight_buf);
        ctx->weight_buf = nullptr;
    }
    if (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }
    // backend_cpu is the global g_cpu_backend — process-wide, do not free.
    // backend can be a per-context Metal handle (from init_best); free that.
    if (ctx->backend && ctx->backend != ctx->backend_cpu) {
        ggml_backend_free(ctx->backend);
    }
    ctx->backend = nullptr;
    delete ctx;
}

void voxcpm2_set_n_threads(struct voxcpm2_context* ctx, int n_threads) {
    if (ctx)
        ctx->n_threads = n_threads > 0 ? n_threads : 1;
}

void voxcpm2_set_seed(struct voxcpm2_context* ctx, uint32_t seed) {
    if (ctx)
        ctx->seed = seed;
}

float* voxcpm2_synthesize(struct voxcpm2_context* ctx, const char* text, int* out_n_samples) {
    return vox_synthesize_internal(ctx, text, nullptr, 0, out_n_samples);
}

float* voxcpm2_synthesize_clone(struct voxcpm2_context* ctx, const char* text, const float* ref_samples,
                                int ref_n_samples, int* out_n_samples) {
    return vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, out_n_samples);
}

struct voxcpm2_stream* voxcpm2_stream_open(struct voxcpm2_context* ctx, const char* text, const float* ref_samples,
                                           int ref_n_samples) {
    if (!ctx || !text)
        return nullptr;
    auto* s = new voxcpm2_stream();
    s->ctx = ctx;
    s->done = false;
    s->chunk_offset = 0;

    int n_samples = 0;
    float* pcm = vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, &n_samples);
    if (pcm && n_samples > 0) {
        s->all_pcm.assign(pcm, pcm + n_samples);
        std::free(pcm);
    } else {
        s->done = true;
    }
    return s;
}

const float* voxcpm2_stream_next(struct voxcpm2_stream* stream, int* out_n_samples) {
    if (!stream || stream->done || stream->all_pcm.empty()) {
        if (out_n_samples)
            *out_n_samples = 0;
        return nullptr;
    }

    const int chunk_size = 3840; // one VAE decode unit at 48kHz (~12.5fps)
    int offset = stream->chunk_offset;
    int remaining = (int)stream->all_pcm.size() - offset;
    if (remaining <= 0) {
        stream->done = true;
        if (out_n_samples)
            *out_n_samples = 0;
        return nullptr;
    }

    int n = std::min(remaining, chunk_size);
    stream->chunk_buf.assign(stream->all_pcm.begin() + offset, stream->all_pcm.begin() + offset + n);
    stream->chunk_offset += n;
    if (stream->chunk_offset >= (int)stream->all_pcm.size()) {
        stream->done = true;
    }

    if (out_n_samples)
        *out_n_samples = n;
    return stream->chunk_buf.data();
}

void voxcpm2_stream_close(struct voxcpm2_stream* stream) {
    delete stream;
}

void voxcpm2_pcm_free(float* pcm) {
    std::free(pcm);
}

float* voxcpm2_extract_stage(struct voxcpm2_context* ctx, const char* text, const float* ref_samples, int ref_n_samples,
                             const char* stage_name, int* out_n) {
    if (!ctx || !text || !stage_name || !out_n)
        return nullptr;
    *out_n = 0;

    ggml_backend_t cpu_be = get_cpu_backend();
    if (ggml_backend_is_cpu(cpu_be)) {
        ggml_backend_cpu_set_n_threads(cpu_be, ctx->n_threads);
    }

    std::string stage(stage_name);
    std::vector<int32_t> token_ids = vox_tokenize(ctx->tokenizer, std::string(text));

    if (stage == "text_input_ids") {
        // Python's dump captures `model.text_tokenizer(syn_text)` — text-only,
        // without the audio_start_token that `vox_tokenize` appends. Strip the
        // trailing audio_start so the count matches.
        std::vector<int32_t> text_only = token_ids;
        if (!text_only.empty() && text_only.back() == (int32_t)ctx->hp.audio_start_token) {
            text_only.pop_back();
        }
        *out_n = (int)text_only.size();
        float* out = (float*)std::malloc((size_t)*out_n * sizeof(float));
        for (int i = 0; i < *out_n; i++) {
            out[i] = (float)text_only[i];
        }
        return out;
    }

    if (stage == "tslm_prefill_out" || stage == "tslm_layer_0_out" || stage == "tslm_layer_27_out") {
        // Instrumented prefill: capture per-position + per-layer outputs.
        // When VOXCPM2_USE_REF=1 and ref_samples is a 16 kHz mono PCM WAV, we
        // run the cloning prefill (ref prefix + text) to match the Python
        // dumper's `_run_prefill` with `use_ref=True`. Otherwise zero-shot.
        int d = (int)ctx->hp.tslm_d_model;
        int n_layers = (int)ctx->hp.tslm_n_layers;
        const int N_CAP = 8; // first 8 positions (matching Python reference)

        std::vector<float> all_pos;
        std::vector<float> layer0_buf, layer_last_buf;

        tslm_prefill_hooks hooks;
        hooks.max_capture_positions = N_CAP;
        hooks.all_positions = &all_pos;
        hooks.layer0_capture = 0;
        hooks.layer0_out = &layer0_buf;
        hooks.layer_last_capture = n_layers - 1;
        hooks.layer_last_out = &layer_last_buf;

        const char* use_ref_env = std::getenv("VOXCPM2_USE_REF");
        bool use_ref = (use_ref_env && std::atoi(use_ref_env) != 0 && ref_samples && ref_n_samples > 0);

        std::vector<uint8_t> audio_mask_ref;
        if (use_ref) {
            const vox_prefill_inputs& pi =
                build_prefill_inputs_cached(ctx, std::string(text), ref_samples, ref_n_samples, cpu_be);
            if (pi.N_pos == 0) {
                return nullptr;
            }
            audio_mask_ref = pi.audio_mask_pos;
            tslm_prefill_from_embeds(ctx, pi.combined_embed.data(), pi.N_pos, cpu_be, hooks);
        } else {
            tslm_prefill_ex(ctx, token_ids, cpu_be, hooks);
        }

        if (stage == "tslm_prefill_out") {
            // Apply output norm to each captured position; under cloning, also
            // apply FSQ to audio positions (Python:
            //   enc_outputs = fsq(enc_outputs) * audio_mask + enc_outputs * text_mask).
            int N = (int)(all_pos.size() / d);
            std::vector<float> normed((size_t)N * d);
            for (int i = 0; i < N; i++) {
                rms_norm_cpu(all_pos.data() + (size_t)i * d, tensor_data_f32(ctx->weights.tslm_output_norm),
                             normed.data() + (size_t)i * d, d, ctx->hp.rms_norm_eps);
            }
            if (use_ref) {
                for (int i = 0; i < N; i++) {
                    if (i < (int)audio_mask_ref.size() && audio_mask_ref[i]) {
                        std::vector<float> fsq_pos = fsq_forward(ctx, normed.data() + (size_t)i * d, cpu_be);
                        std::memcpy(normed.data() + (size_t)i * d, fsq_pos.data(), (size_t)d * sizeof(float));
                    }
                }
            }
            *out_n = N * d;
            float* out = (float*)std::malloc((size_t)*out_n * sizeof(float));
            std::memcpy(out, normed.data(), (size_t)*out_n * sizeof(float));
            return out;
        } else if (stage == "tslm_layer_0_out") {
            *out_n = (int)layer0_buf.size();
            if (*out_n == 0)
                return nullptr;
            float* out = (float*)std::malloc(layer0_buf.size() * sizeof(float));
            std::memcpy(out, layer0_buf.data(), layer0_buf.size() * sizeof(float));
            return out;
        } else { // tslm_layer_27_out
            *out_n = (int)layer_last_buf.size();
            if (*out_n == 0)
                return nullptr;
            float* out = (float*)std::malloc(layer_last_buf.size() * sizeof(float));
            std::memcpy(out, layer_last_buf.data(), layer_last_buf.size() * sizeof(float));
            return out;
        }
    }

    if (stage == "ralm_prefill_out") {
        // Multi-position RALM prefill matching Python:
        // 1. (TSLM prefill | combined-embed prefill for cloning) → normed
        // 2. FSQ masking: audio positions get FSQ-quantised output
        // 3. fusion_concat_proj(cat(enc_out, audio_mask * feat_embed))
        // 4. RALM prefill (causal, all positions)
        // 5. Capture first 8 positions
        const int N_CAP = 8;
        int d = (int)ctx->hp.tslm_d_model;

        const char* use_ref_env = std::getenv("VOXCPM2_USE_REF");
        bool use_ref = (use_ref_env && std::atoi(use_ref_env) != 0 && ref_samples && ref_n_samples > 0);

        std::vector<float> all_pos;
        std::vector<uint8_t> audio_mask_ref;
        std::vector<float> feat_embed_ref;
        tslm_prefill_hooks hooks;
        hooks.max_capture_positions = N_CAP;
        hooks.all_positions = &all_pos;

        if (use_ref) {
            const vox_prefill_inputs& pi =
                build_prefill_inputs_cached(ctx, std::string(text), ref_samples, ref_n_samples, cpu_be);
            if (pi.N_pos == 0) {
                return nullptr;
            }
            audio_mask_ref = pi.audio_mask_pos;
            feat_embed_ref = pi.feat_embed_pos;
            tslm_prefill_from_embeds(ctx, pi.combined_embed.data(), pi.N_pos, cpu_be, hooks);
        } else {
            tslm_prefill_ex(ctx, token_ids, cpu_be, hooks);
        }

        int N = (int)(all_pos.size() / d);
        std::vector<float> normed((size_t)N * d);
        for (int i = 0; i < N; i++) {
            rms_norm_cpu(all_pos.data() + (size_t)i * d, tensor_data_f32(ctx->weights.tslm_output_norm),
                         normed.data() + (size_t)i * d, d, ctx->hp.rms_norm_eps);
        }
        if (use_ref) {
            for (int i = 0; i < N; i++) {
                if (i < (int)audio_mask_ref.size() && audio_mask_ref[i]) {
                    std::vector<float> fsq_pos = fsq_forward(ctx, normed.data() + (size_t)i * d, cpu_be);
                    std::memcpy(normed.data() + (size_t)i * d, fsq_pos.data(), (size_t)d * sizeof(float));
                }
            }
        }

        // fusion_concat_proj: cat(enc_out, audio_mask * feat_embed) per position
        int in_dim = 2 * d;
        std::vector<float> ralm_input((size_t)N * d);
        for (int i = 0; i < N; i++) {
            std::vector<float> cat_buf(in_dim, 0.0f);
            std::memcpy(cat_buf.data(), normed.data() + (size_t)i * d, (size_t)d * sizeof(float));
            if (use_ref && i < (int)audio_mask_ref.size() && audio_mask_ref[i]) {
                std::memcpy(cat_buf.data() + d, feat_embed_ref.data() + (size_t)i * d, (size_t)d * sizeof(float));
            }
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, cat_buf.data(), in_dim,
                           ralm_input.data() + (size_t)i * d, d);
        }

        std::vector<float> ralm_out = ralm_prefill_multi(ctx, ralm_input.data(), N, cpu_be);
        for (int i = 0; i < N; i++) {
            rms_norm_cpu(ralm_out.data() + (size_t)i * d, tensor_data_f32(ctx->weights.ralm_output_norm),
                         ralm_out.data() + (size_t)i * d, d, ctx->hp.rms_norm_eps);
        }

        *out_n = N * d;
        float* out = (float*)std::malloc((size_t)*out_n * sizeof(float));
        std::memcpy(out, ralm_out.data(), (size_t)*out_n * sizeof(float));
        return out;
    }

    if (stage == "dit_single_fwd") {
        // Single LocDiT forward pass with reference input sequence.
        // ref_samples should contain dit_input_seq [T_seq * d_dit] = [11 * 1024] floats.
        // Runs 12 transformer layers + final norm + out_proj, returns velocity [C=64, T=4].
        int d_dit = (int)ctx->hp.locdit_d_model;
        int P_fr = (int)ctx->hp.patch_frames;
        int feat_dim = 64;
        int mu_toks = 2;
        int T_seq = mu_toks + 1 + P_fr + P_fr; // 11
        int expected_ref = T_seq * d_dit;      // 11264

        if (!ref_samples || ref_n_samples < expected_ref) {
            fprintf(stderr, "dit_single_fwd: need dit_input_seq reference (%d floats, got %d)\n", expected_ref,
                    ref_n_samples);
            return nullptr;
        }

        // Feed the reference sequence through 12 LocDiT layers + norm + out_proj
        const vox_hparams& hp2 = ctx->hp;
        const vox_weights& W2 = ctx->weights;
        int d = d_dit;
        int n_q = (int)hp2.locdit_n_heads;
        int n_kv = (int)hp2.locdit_n_kv;
        int hd = (int)hp2.locdit_head_dim;
        float eps = hp2.rms_norm_eps;
        float ascale = 1.0f / std::sqrt((float)hd);
        int x_offset = mu_toks + 1 + P_fr; // 7

        // Copy reference sequence into working buffer
        std::vector<float> cur((size_t)T_seq * d);
        std::memcpy(cur.data(), ref_samples, (size_t)expected_ref * sizeof(float));

        // LongRoPE factors
        const float* rope_factors = W2.tslm_rope_short ? tensor_data_f32(W2.tslm_rope_short) : nullptr;
        float rope_theta = hp2.tslm_rope_theta;

        std::vector<float> normed((size_t)T_seq * d), attn_out((size_t)T_seq * d), ffn_h(d);

        for (int l = 0; l < (int)hp2.locdit_n_layers; l++) {
            const vox_enc_layer& L = W2.locdit_layers[l];
            for (int t = 0; t < T_seq; t++) {
                rms_norm_cpu(cur.data() + (size_t)t * d, tensor_data_f32(L.norm1_w), normed.data() + (size_t)t * d, d,
                             eps);
            }
            bidir_attn_full(normed.data(), T_seq, d, L.attn_q_w, L.attn_k_w, L.attn_v_w, L.attn_o_w, n_q, n_kv, hd,
                            ascale, cpu_be, attn_out.data(), rope_factors, rope_theta);
            for (size_t i = 0; i < (size_t)T_seq * d; i++)
                cur[i] += attn_out[i];
            for (int t = 0; t < T_seq; t++) {
                rms_norm_cpu(cur.data() + (size_t)t * d, tensor_data_f32(L.norm2_w), normed.data() + (size_t)t * d, d,
                             eps);
                swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data() + (size_t)t * d, d,
                               (int)hp2.locdit_ff_dim, d, ffn_h.data());
                float* ct = cur.data() + (size_t)t * d;
                for (int i = 0; i < d; i++)
                    ct[i] += ffn_h[i];
            }
        }

        // Extract x-portion (last P_fr tokens) → final norm → out_proj
        int out_size = feat_dim * P_fr;
        std::vector<float> vel(out_size);
        for (int p = 0; p < P_fr; p++) {
            const float* h_p = cur.data() + (size_t)(x_offset + p) * d;
            float* v_p = vel.data() + (size_t)p * feat_dim;
            std::vector<float> normed_p(d);
            if (W2.locdit_norm_w) {
                rms_norm_cpu(h_p, tensor_data_f32(W2.locdit_norm_w), normed_p.data(), d, eps);
            } else {
                std::memcpy(normed_p.data(), h_p, (size_t)d * sizeof(float));
            }
            if (W2.locdit_out_proj_w && W2.locdit_out_proj_b) {
                matmul_mv_bias(cpu_be, W2.locdit_out_proj_w, W2.locdit_out_proj_b, normed_p.data(), d, v_p, feat_dim);
            }
        }

        // Output in time-first [T=4, C=64] layout (matching Python vel[0] which is [C=64, T=4])
        // Actually Python vel_sf[0] is shape [C=64, T=4] (channels-first from estimator output)
        // We produce [P_fr * feat_dim] in time-first layout → need to transpose to channels-first
        std::vector<float> vel_cf(out_size);
        for (int t = 0; t < P_fr; t++)
            for (int c = 0; c < feat_dim; c++)
                vel_cf[c * P_fr + t] = vel[t * feat_dim + c];

        *out_n = out_size;
        float* out = (float*)std::malloc((size_t)out_size * sizeof(float));
        std::memcpy(out, vel_cf.data(), (size_t)out_size * sizeof(float));
        return out;
    }

    if (stage == "cfm_step0_result") {
        // CFM Euler solve using reference mu + noise when available.
        // ref_samples layout (packed by diff handler):
        //   [mu_data (2*d_dit floats), noise_data (state_size floats)]
        // If ref has both cfm_mu and cfm_step0_z, uses exact Python values.
        // Otherwise computes mu from TSLM+RALM and uses seeded F32 noise.
        int d = (int)ctx->hp.tslm_d_model;
        int d_dit = (int)ctx->hp.locdit_d_model;
        int d_ralm = (int)ctx->hp.ralm_d_model;
        int P_fr = (int)ctx->hp.patch_frames;
        int state_size = 64 * P_fr;
        int mu_size = 2 * d_dit;

        std::vector<float> mu(mu_size, 0.0f);
        std::vector<float> noise(state_size);
        bool have_ref_mu = (ref_samples && ref_n_samples >= mu_size + state_size);

        if (have_ref_mu) {
            // Unpack reference mu + noise (exact Python values)
            std::memcpy(mu.data(), ref_samples, (size_t)mu_size * sizeof(float));
            std::memcpy(noise.data(), ref_samples + mu_size, (size_t)state_size * sizeof(float));
        } else {
            // Compute mu from scratch (F16/F32 precision gap expected)
            std::vector<float> tslm_h = tslm_prefill(ctx, token_ids, cpu_be);
            {
                std::vector<float> normed(d);
                rms_norm_cpu(tslm_h.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d,
                             ctx->hp.rms_norm_eps);
                tslm_h = normed;
            }
            int in_dim = 2 * d;
            std::vector<float> cat_buf(in_dim, 0.0f);
            std::memcpy(cat_buf.data(), tslm_h.data(), (size_t)d * sizeof(float));
            std::vector<float> ralm_input(d);
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, cat_buf.data(), in_dim,
                           ralm_input.data(), d);
            std::vector<float> ralm_h = ralm_prefill(ctx, ralm_input, cpu_be);
            {
                std::vector<float> normed(d_ralm);
                rms_norm_cpu(ralm_h.data(), tensor_data_f32(ctx->weights.ralm_output_norm), normed.data(), d_ralm,
                             ctx->hp.rms_norm_eps);
                ralm_h = normed;
            }
            if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b)
                matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b, tslm_h.data(), d, mu.data(),
                               d_dit);
            if (ctx->weights.res_to_dit_w && ctx->weights.res_to_dit_b)
                matmul_mv_bias(cpu_be, ctx->weights.res_to_dit_w, ctx->weights.res_to_dit_b, ralm_h.data(), d_ralm,
                               mu.data() + d_dit, d_dit);

            // Use ref noise if provided (just noise, no mu)
            if (ref_samples && ref_n_samples >= state_size) {
                std::memcpy(noise.data(), ref_samples, (size_t)state_size * sizeof(float));
            } else {
                fill_gaussian_noise(noise.data(), state_size, 42);
            }
        }

        std::vector<float> zero_cond(state_size, 0.0f);
        std::vector<float> patch =
            cfm_euler_solve(ctx, mu.data(), zero_cond.data(), 10, ctx->cfg_value, cpu_be, noise.data());
        // CFM returns channels-first [C=64, T=4]. Python reference saves time-first [T=4, C=64]
        // (pred_feat = cfm(...).transpose(1, 2)), so transpose to match.
        int P_tf = P_fr, C_tf = 64;
        std::vector<float> patch_tf(state_size);
        for (int t = 0; t < P_tf; t++)
            for (int c = 0; c < C_tf; c++)
                patch_tf[t * C_tf + c] = patch[c * P_tf + t];
        *out_n = (int)patch_tf.size();
        float* out = (float*)std::malloc(patch_tf.size() * sizeof(float));
        std::memcpy(out, patch_tf.data(), patch_tf.size() * sizeof(float));
        return out;
    }

    if (stage == "locenc_out") {
        // Python: feat_embed = model.feat_encoder(audio_feat) → [B, T, d_enc]
        // For zero-shot: audio_feat is zeros [T, P, 64] → all positions identical
        //   (compute LocEnc(0) once, replicate).
        // For cloning (VOXCPM2_USE_REF=1): audio_feat = [z1, ref_feat..., z1, 0..].
        //   Run LocEnc per-position for the first N_CAP positions.
        int d_enc = (int)ctx->hp.locenc_d_model; // 1024
        int P_fr = (int)ctx->hp.patch_frames;
        int feat_dim = 64;
        const int N_CAP = 8;
        int total = N_CAP * d_enc;
        *out_n = total;
        float* out = (float*)std::malloc((size_t)total * sizeof(float));

        const char* use_ref_env = std::getenv("VOXCPM2_USE_REF");
        bool use_ref = (use_ref_env && std::atoi(use_ref_env) != 0 && ref_samples && ref_n_samples > 0);

        if (use_ref) {
            // Direct VAE encode — locenc_out only needs the ref patches, not the
            // full TSLM prefill state.
            int T_ref = 0;
            const std::vector<float>& ref_feat = vae_encode_cached(ctx, ref_samples, ref_n_samples, &T_ref);
            std::vector<float> z_patch((size_t)feat_dim * P_fr, 0.0f);
            for (int i = 0; i < N_CAP; i++) {
                const float* patch_ptr = z_patch.data();
                if (i >= 1 && i <= T_ref) {
                    patch_ptr = ref_feat.data() + (size_t)(i - 1) * P_fr * feat_dim;
                }
                std::vector<float> enc = locenc_forward(ctx, patch_ptr, cpu_be);
                std::memcpy(out + (size_t)i * d_enc, enc.data(), (size_t)d_enc * sizeof(float));
            }
        } else {
            std::vector<float> zero_patch((size_t)feat_dim * P_fr, 0.0f);
            std::vector<float> one_out = locenc_forward(ctx, zero_patch.data(), cpu_be);
            for (int i = 0; i < N_CAP; i++) {
                std::memcpy(out + (size_t)i * d_enc, one_out.data(), (size_t)d_enc * sizeof(float));
            }
        }
        return out;
    }

    if (stage == "locenc_in") {
        // LocEnc input: audio_feat[0, :8] = [8, patch_frames, feat_dim] = [8, 4, 64].
        // Zero-shot: all zeros. Cloning: positions 1..T_ref hold encoded ref_feat,
        // positions 0 and T_ref+1 are z1 zeros, rest are text_pad zeros.
        int P_fr = (int)ctx->hp.patch_frames;
        int feat_dim = 64;
        int N_CAP = 8;
        int total = N_CAP * P_fr * feat_dim;
        *out_n = total;
        float* out = (float*)std::calloc(total, sizeof(float));

        const char* use_ref_env = std::getenv("VOXCPM2_USE_REF");
        bool use_ref = (use_ref_env && std::atoi(use_ref_env) != 0 && ref_samples && ref_n_samples > 0);
        if (use_ref) {
            // Direct VAE encode — no need for full prefill state for locenc_in.
            int T_ref = 0;
            const std::vector<float>& ref_feat = vae_encode_cached(ctx, ref_samples, ref_n_samples, &T_ref);
            for (int i = 0; i < N_CAP; i++) {
                if (i >= 1 && i <= T_ref) {
                    std::memcpy(out + (size_t)i * P_fr * feat_dim, ref_feat.data() + (size_t)(i - 1) * P_fr * feat_dim,
                                (size_t)P_fr * feat_dim * sizeof(float));
                }
            }
        }
        return out;
    }

    if (stage == "enc_to_lm") {
        // Python: feat_embed = enc_to_lm_proj(feat_encoder(audio_feat)) → [B, T, d_lm].
        // Same per-position semantics as locenc_out.
        int d_enc = (int)ctx->hp.locenc_d_model; // 1024
        int d_lm = (int)ctx->hp.tslm_d_model;    // 2048
        int P_fr = (int)ctx->hp.patch_frames;
        int feat_dim = 64;
        const int N_CAP = 8;
        int total = N_CAP * d_lm;
        if (!ctx->weights.enc_to_lm_w || !ctx->weights.enc_to_lm_b) {
            return nullptr;
        }
        *out_n = total;
        float* out = (float*)std::malloc((size_t)total * sizeof(float));

        const char* use_ref_env = std::getenv("VOXCPM2_USE_REF");
        bool use_ref = (use_ref_env && std::atoi(use_ref_env) != 0 && ref_samples && ref_n_samples > 0);

        if (use_ref) {
            // Direct VAE encode + LocEnc + enc_to_lm per position. Same input
            // dependency as locenc_out; no need for full TSLM prefill state.
            int T_ref = 0;
            const std::vector<float>& ref_feat = vae_encode_cached(ctx, ref_samples, ref_n_samples, &T_ref);
            std::vector<float> z_patch((size_t)feat_dim * P_fr, 0.0f);
            std::vector<float> proj(d_lm);
            for (int i = 0; i < N_CAP; i++) {
                const float* patch_ptr = z_patch.data();
                if (i >= 1 && i <= T_ref) {
                    patch_ptr = ref_feat.data() + (size_t)(i - 1) * P_fr * feat_dim;
                }
                std::vector<float> enc = locenc_forward(ctx, patch_ptr, cpu_be);
                matmul_mv_bias(cpu_be, ctx->weights.enc_to_lm_w, ctx->weights.enc_to_lm_b, enc.data(), d_enc,
                               proj.data(), d_lm);
                std::memcpy(out + (size_t)i * d_lm, proj.data(), (size_t)d_lm * sizeof(float));
            }
            return out;
        } else {
            std::vector<float> zero_patch((size_t)feat_dim * P_fr, 0.0f);
            std::vector<float> enc_out = locenc_forward(ctx, zero_patch.data(), cpu_be);
            std::vector<float> proj_one(d_lm);
            matmul_mv_bias(cpu_be, ctx->weights.enc_to_lm_w, ctx->weights.enc_to_lm_b, enc_out.data(), d_enc,
                           proj_one.data(), d_lm);
            for (int i = 0; i < N_CAP; i++) {
                std::memcpy(out + (size_t)i * d_lm, proj_one.data(), (size_t)d_lm * sizeof(float));
            }
            return out;
        }
        return nullptr;
    }

    if (stage == "tslm_last_hidden") {
        // Raw TSLM last-token output (normed, before projection) — for debugging
        std::vector<float> h = tslm_prefill(ctx, token_ids, cpu_be);
        int d = (int)ctx->hp.tslm_d_model;
        std::vector<float> normed(d);
        rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d, ctx->hp.rms_norm_eps);
        *out_n = d;
        float* out = (float*)std::malloc((size_t)d * sizeof(float));
        std::memcpy(out, normed.data(), (size_t)d * sizeof(float));
        return out;
    }

    if (stage == "lm_to_dit_hidden" || stage == "res_to_dit_hidden") {
        // Run full pipeline: prefill (TSLM all positions) → norm → FSQ → fusion → RALM (multi) → project.
        // Under VOXCPM2_USE_REF=1 the prefill uses the cloning combined-embed (ref
        // prefix + text), FSQ at ref-audio positions, and feat_embed in the fusion
        // concat — matching Python `_inference` with use_ref=True. Otherwise zero-shot.
        int d = (int)ctx->hp.tslm_d_model;
        int d_dit = (int)ctx->hp.locdit_d_model;
        int T_tok = (int)token_ids.size();

        const char* use_ref_env = std::getenv("VOXCPM2_USE_REF");
        bool use_ref = (use_ref_env && std::atoi(use_ref_env) != 0 && ref_samples && ref_n_samples > 0);

        std::vector<float> all_pos;
        std::vector<uint8_t> audio_mask_ref;
        std::vector<float> feat_embed_ref;
        tslm_prefill_hooks hooks;
        hooks.max_capture_positions = use_ref ? 0 : T_tok;
        hooks.all_positions = &all_pos;

        if (use_ref) {
            const vox_prefill_inputs& pi =
                build_prefill_inputs_cached(ctx, std::string(text), ref_samples, ref_n_samples, cpu_be);
            if (pi.N_pos == 0) {
                return nullptr;
            }
            hooks.max_capture_positions = pi.N_pos;
            audio_mask_ref = pi.audio_mask_pos;
            feat_embed_ref = pi.feat_embed_pos;
            tslm_prefill_from_embeds(ctx, pi.combined_embed.data(), pi.N_pos, cpu_be, hooks);
        } else {
            tslm_prefill_ex(ctx, token_ids, cpu_be, hooks);
        }
        int N = (int)(all_pos.size() / d);

        // Apply output norm to each position
        std::vector<float> normed_all((size_t)N * d);
        for (int i = 0; i < N; i++) {
            rms_norm_cpu(all_pos.data() + (size_t)i * d, tensor_data_f32(ctx->weights.tslm_output_norm),
                         normed_all.data() + (size_t)i * d, d, ctx->hp.rms_norm_eps);
        }

        // FSQ masking at ref-audio positions (Python:
        //   enc_outputs = fsq(enc) * audio_mask + enc * text_mask).
        if (use_ref) {
            for (int i = 0; i < N; i++) {
                if (i < (int)audio_mask_ref.size() && audio_mask_ref[i]) {
                    std::vector<float> fsq_pos = fsq_forward(ctx, normed_all.data() + (size_t)i * d, cpu_be);
                    std::memcpy(normed_all.data() + (size_t)i * d, fsq_pos.data(), (size_t)d * sizeof(float));
                }
            }
        }

        // Last-token TSLM output (for lm_to_dit)
        std::vector<float> tslm_out(normed_all.data() + (size_t)(N - 1) * d, normed_all.data() + (size_t)N * d);

        // fusion_concat_proj: cat(enc_out, audio_mask * feat_embed). For zero-shot
        // every audio_mask is 0 so the second half is all zeros.
        int in_dim = 2 * d;
        std::vector<float> ralm_input((size_t)N * d);
        for (int i = 0; i < N; i++) {
            std::vector<float> cat_buf(in_dim, 0.0f);
            std::memcpy(cat_buf.data(), normed_all.data() + (size_t)i * d, (size_t)d * sizeof(float));
            if (use_ref && i < (int)audio_mask_ref.size() && audio_mask_ref[i]) {
                std::memcpy(cat_buf.data() + d, feat_embed_ref.data() + (size_t)i * d, (size_t)d * sizeof(float));
            }
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, cat_buf.data(), in_dim,
                           ralm_input.data() + (size_t)i * d, d);
        }

        // Multi-position RALM prefill (causal, all T positions)
        std::vector<float> ralm_out = ralm_prefill_multi(ctx, ralm_input.data(), N, cpu_be);

        // Apply RALM output norm and extract last position
        int dr = (int)ctx->hp.ralm_d_model;
        std::vector<float> ralm_h(dr);
        rms_norm_cpu(ralm_out.data() + (size_t)(N - 1) * dr, tensor_data_f32(ctx->weights.ralm_output_norm),
                     ralm_h.data(), dr, ctx->hp.rms_norm_eps);

        if (stage == "lm_to_dit_hidden") {
            if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b) {
                std::vector<float> proj(d_dit);
                matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b, tslm_out.data(), d,
                               proj.data(), d_dit);
                *out_n = d_dit;
                float* out = (float*)std::malloc((size_t)d_dit * sizeof(float));
                std::memcpy(out, proj.data(), (size_t)d_dit * sizeof(float));
                return out;
            }
        } else { // res_to_dit_hidden
            if (ctx->weights.res_to_dit_w && ctx->weights.res_to_dit_b) {
                int d_ralm = (int)ctx->hp.ralm_d_model;
                std::vector<float> proj(d_dit);
                matmul_mv_bias(cpu_be, ctx->weights.res_to_dit_w, ctx->weights.res_to_dit_b, ralm_h.data(), d_ralm,
                               proj.data(), d_dit);
                *out_n = d_dit;
                float* out = (float*)std::malloc((size_t)d_dit * sizeof(float));
                std::memcpy(out, proj.data(), (size_t)d_dit * sizeof(float));
                return out;
            }
        }
        return nullptr;
    }

    if (stage == "stop_logits_step0") {
        // Run TSLM prefill, then compute stop logits from the last hidden state
        std::vector<float> h = tslm_prefill(ctx, token_ids, cpu_be);
        int d = (int)ctx->hp.tslm_d_model;
        {
            std::vector<float> normed(d);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d,
                         ctx->hp.rms_norm_eps);
            h = normed;
        }
        // stop_proj: d_lm → d_lm with bias, then SiLU, then stop_head: d_lm → 2
        const vox_weights& W = ctx->weights;
        if (!W.stop_proj_w || !W.stop_proj_b || !W.stop_head_w)
            return nullptr;

        std::vector<float> proj(d);
        matmul_mv_bias(cpu_be, W.stop_proj_w, W.stop_proj_b, h.data(), d, proj.data(), d);
        for (int i = 0; i < d; i++) {
            float v = proj[i];
            proj[i] = v / (1.0f + std::exp(-v)); // SiLU
        }
        float logits[2] = {0.0f, 0.0f};
        matmul_mv(cpu_be, W.stop_head_w, proj.data(), d, logits, 2);

        *out_n = 2;
        float* out = (float*)std::malloc(2 * sizeof(float));
        out[0] = logits[0];
        out[1] = logits[1];
        return out;
    }

    if (stage == "dit_input_seq") {
        // Build the LocDiT input sequence [T=11, d=1024] matching locdit_forward
        // Uses reference mu (from ref_samples) if available for exact conditioning.
        int d_dit = (int)ctx->hp.locdit_d_model; // 1024
        int P_fr = (int)ctx->hp.patch_frames;    // 4
        int feat_dim = 64;
        int mu_size = 2 * d_dit; // 2048
        int state_size = feat_dim * P_fr;
        int mu_toks = 2, t_tok = 1;
        int T_seq = mu_toks + t_tok + P_fr + P_fr; // 11

        // Get mu and noise from ref_samples if packed [mu..., noise...]
        const float* mu_data = nullptr;
        const float* noise_data = nullptr;
        if (ref_samples && ref_n_samples >= mu_size + state_size) {
            mu_data = ref_samples;
            noise_data = ref_samples + mu_size;
        }

        // Build sequence
        std::vector<float> seq((size_t)T_seq * d_dit, 0.0f);

        // Tokens 0-1: mu [2048] split into 2 x [1024]
        if (mu_data) {
            std::memcpy(seq.data(), mu_data, (size_t)d_dit * sizeof(float));
            std::memcpy(seq.data() + d_dit, mu_data + d_dit, (size_t)d_dit * sizeof(float));
        }

        // Token 2: time embedding (sinusoidal → MLP)
        // t value at sway step 2 (first real LocDiT step). Compute in bf16 to
        // match Python's BF16 t_span (linspace+sway both run in bf16).
        float t_raw = bf16_round(1.0f - 1.0f / 10.0f); // bf16(0.9)
        float a = bf16_round((float)M_PI / 2.0f * t_raw);
        float cos_a = bf16_round(std::cos(a));
        float c = bf16_round(cos_a - 1.0f);
        float d_term = bf16_round(c + t_raw);
        float t_val = bf16_round(t_raw + d_term);
        auto t_sin = sinusoidal_time_emb(t_val, d_dit);
        const vox_weights& W = ctx->weights;
        std::vector<float> t_emb(d_dit), dt_emb(d_dit);
        // time MLP
        if (W.locdit_time_mlp_0_w && W.locdit_time_mlp_1_w) {
            std::vector<float> h0(d_dit);
            matmul_mv_bias(cpu_be, W.locdit_time_mlp_0_w, W.locdit_time_mlp_0_b, t_sin.data(), d_dit, h0.data(), d_dit);
            for (int i = 0; i < d_dit; i++)
                h0[i] = h0[i] / (1.0f + std::exp(-h0[i]));
            matmul_mv_bias(cpu_be, W.locdit_time_mlp_1_w, W.locdit_time_mlp_1_b, h0.data(), d_dit, t_emb.data(), d_dit);
        }
        // dt MLP (dt=0)
        auto dt_sin = sinusoidal_time_emb(0.0f, d_dit);
        if (W.locdit_dt_mlp_0_w && W.locdit_dt_mlp_1_w) {
            std::vector<float> h0(d_dit);
            matmul_mv_bias(cpu_be, W.locdit_dt_mlp_0_w, W.locdit_dt_mlp_0_b, dt_sin.data(), d_dit, h0.data(), d_dit);
            for (int i = 0; i < d_dit; i++)
                h0[i] = h0[i] / (1.0f + std::exp(-h0[i]));
            matmul_mv_bias(cpu_be, W.locdit_dt_mlp_1_w, W.locdit_dt_mlp_1_b, h0.data(), d_dit, dt_emb.data(), d_dit);
        }
        for (int i = 0; i < d_dit; i++)
            t_emb[i] += dt_emb[i];
        std::memcpy(seq.data() + (size_t)2 * d_dit, t_emb.data(), (size_t)d_dit * sizeof(float));

        // Tokens 3-6: cond_proj (zeros for first step)
        std::vector<float> zero_feat(feat_dim, 0.0f);
        for (int p = 0; p < P_fr; p++) {
            float* dst = seq.data() + (size_t)(mu_toks + t_tok + p) * d_dit;
            if (W.locdit_cond_proj_w && W.locdit_cond_proj_b) {
                matmul_mv_bias(cpu_be, W.locdit_cond_proj_w, W.locdit_cond_proj_b, zero_feat.data(), feat_dim, dst,
                               d_dit);
            }
        }

        // Tokens 7-10: in_proj applied to noise frames
        if (noise_data) {
            for (int p = 0; p < P_fr; p++) {
                float* dst = seq.data() + (size_t)(mu_toks + t_tok + P_fr + p) * d_dit;
                // Noise is [C=64, T=4] in GGUF: data[c*4 + t]. Frame p = data[c*4 + p] for c=0..63
                std::vector<float> frame(feat_dim);
                for (int c = 0; c < feat_dim; c++)
                    frame[c] = noise_data[c * P_fr + p];
                if (W.locdit_in_proj_w && W.locdit_in_proj_b) {
                    matmul_mv_bias(cpu_be, W.locdit_in_proj_w, W.locdit_in_proj_b, frame.data(), feat_dim, dst, d_dit);
                }
            }
        }

        *out_n = T_seq * d_dit;
        float* out = (float*)std::malloc((size_t)*out_n * sizeof(float));
        std::memcpy(out, seq.data(), (size_t)*out_n * sizeof(float));
        return out;
    }

    if (stage == "cfm_step0_z") {
        // CFM initial noise. The Python reference uses torch.randn with
        // dtype=bfloat16 and seed=42 via a separate Generator. The BF16 randn
        // path in PyTorch uses a different bit-extraction than F32 randn, making
        // exact matching non-trivial. For now, output seeded F32 Gaussian noise
        // (matching the MT19937 F32 path used in vibevoice/chatterbox).
        // TODO: implement PyTorch's BF16 randn path for exact match.
        int P_fr = (int)ctx->hp.patch_frames; // 4
        int in_ch = 64;
        int state_size = in_ch * P_fr; // 256
        *out_n = state_size;
        float* out = (float*)std::malloc((size_t)state_size * sizeof(float));
        fill_gaussian_noise_bf16(out, state_size, 42);
        return out;
    }

    if (stage == "decoded_audio") {
        return vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, out_n);
    }

    // VAE-only path: caller passes Python's `generated_latent` (shape
    // [D=64, T_lat]) via ref_samples. We reshape into patches of P frames
    // each and call vae_decode (legacy or graph) directly. This isolates
    // VAE behaviour from the upstream AR drift (TSLM/RALM/LocDiT/CFM)
    // when validating against Python's decoded_audio reference.
    if (stage == "vae_only" || stage == "vae_only_graph") {
        if (!ref_samples || ref_n_samples <= 0) {
            fprintf(stderr, "voxcpm2_extract_stage: %s needs generated_latent via ref_samples\n", stage_name);
            return nullptr;
        }
        const int feat_dim = 64;
        const int P = (int)ctx->hp.patch_frames;
        const int T_lat = ref_n_samples / feat_dim;
        if (T_lat <= 0 || (T_lat % P) != 0) {
            fprintf(stderr,
                    "voxcpm2_extract_stage: %s expects ref_samples = D*T_lat with T_lat divisible by P (got %d)\n",
                    stage_name, ref_n_samples);
            return nullptr;
        }
        const int n_patches = T_lat / P;
        std::vector<std::vector<float>> patches(n_patches);
        // Python layout is [D=64, T_lat], i.e. ref_samples[d * T_lat + t] = lat[d, t].
        // vae_decode expects patches as [P * feat_dim] = [t_in_patch, d] interleaved.
        // (see vae_decode lines reading `patch[p * feat_dim + c]` then writing
        //  latents[c * T_lat + (n*P + p)]).
        for (int n = 0; n < n_patches; n++) {
            patches[n].resize((size_t)P * feat_dim);
            for (int p = 0; p < P; p++) {
                int t = n * P + p;
                for (int c = 0; c < feat_dim; c++) {
                    patches[n][(size_t)p * feat_dim + c] = ref_samples[(size_t)c * T_lat + t];
                }
            }
        }
        std::vector<float> pcm;
        if (stage == "vae_only_graph") {
            pcm = vae_decode_graph(ctx, patches);
        } else {
            pcm = vae_decode(ctx, patches, ctx->backend_cpu);
        }
        *out_n = (int)pcm.size();
        float* out = (float*)std::malloc(pcm.size() * sizeof(float));
        if (!out) {
            *out_n = 0;
            return nullptr;
        }
        std::memcpy(out, pcm.data(), pcm.size() * sizeof(float));
        (void)text;
        return out;
    }

    fprintf(stderr, "voxcpm2_extract_stage: unknown stage '%s'\n", stage_name);
    return nullptr;
}

} // extern "C"
