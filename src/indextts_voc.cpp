// indextts_voc.cpp -- BigVGAN vocoder for IndexTTS-1.5.
//
// Architecture: BigVGAN v2 with SnakeBeta activations and anti-aliased
// multi-periodicity composition (AMPBlock1). Converts GPT hidden states
// (d=1280, T time steps) to 24 kHz mono waveform.
//
// Forward pass:
//   latent [T, 1280]  (GPT hidden states)
//   x = conv_pre(latent)         Conv1d(1280, 1536, k=7, pad=3)
//   x = x + cond_layer(spk_emb) Conv1d(512, 1536, k=1)
//   for i in 0..5:
//       x = snake_beta(x)
//       x = ups[i](x)            ConvTranspose1d upsample
//       x = x + conds[i](spk_emb)
//       xs = 0
//       for j in 0..2:
//           xs += resblock[i*3+j](x)
//       x = xs / 3
//   x = snake_beta_post(x)
//   x = conv_post(x)             Conv1d(24, 1, k=7, pad=3)
//   x = tanh(x)
//
// SnakeBeta: x + (1/beta) * sin(alpha * x)^2
//   where alpha = exp(log_alpha), beta = exp(log_beta) (per-channel).
//
// Weight-norm is already fused in the GGUF tensors.
// SnakeBeta uses native ggml ops (sin, sqr, mul, div, add) for GPU execution.
// Anti-aliased up/downsampling is omitted: ggml_conv_1d_dw has no CUDA kernel,
// so the raw SnakeBeta (x + sin(αx)²/β) gives full GPU offload.

#include "indextts_voc.h"
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h> // vDSP_conv, vvsinf, vDSP_vsq — Step C-1
#endif

namespace {

// ── Hyperparameters ──────────────────────────────────────────────

struct bigvgan_hp {
    int gpt_dim = 1280;
    int upsample_initial_ch = 1536;
    int num_upsamples = 6;
    int num_kernels = 3;
    int spk_emb_dim = 512;
    int sampling_rate = 24000;
    int hop_size = 256;

    int upsample_rates[6] = {4, 4, 4, 4, 2, 2};
    int upsample_kernel_sizes[6] = {8, 8, 4, 4, 4, 4};
    int resblock_kernel_sizes[3] = {3, 7, 11};
    // Dilations per dilated pass: [1, 3, 5] for all resblocks.
    int resblock_dilations[3] = {1, 3, 5};
};

// ── Tensor lookup helper ────────────────────────────────────────

static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const char* name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

} // namespace

// ── Context ─────────────────────────────────────────────────────

// Anti-aliased SnakeBeta params for the CPU path.
// Pre-scaled filters and per-thread scratch buffers eliminate the per-channel
// std::vector allocations that used to dominate runtime.
struct aa_snake_params {
    std::vector<float> alpha;        // exp(log_alpha), per channel
    std::vector<float> beta;         // exp(log_beta), per channel
    std::vector<float> us_filter_x2; // upsample filter * 2.0f, length K
    std::vector<float> ds_filter;    // downsample filter, length K
    int C;

    // Thread-local scratch (resized lazily on first use).
    // Indexed by ith (worker id); access is single-writer per thread.
    mutable std::vector<std::vector<float>> scratch_padded;
    mutable std::vector<std::vector<float>> scratch_upsampled;
    mutable std::vector<std::vector<float>> scratch_dspadded;
    // Step C-1: separate scratch for the SnakeBeta vector workspace (vvsinf
    // writes one tmp buffer the size of T_cropped — using the upsample tail
    // would overrun whenever T_cropped > up_pad_right, which is the common case).
    mutable std::vector<std::vector<float>> scratch_snake;
};

struct indextts_voc_context {
    bigvgan_hp hp;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Compute scheduler
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    int n_threads = 4;
    int verbosity = 1;
    // BigVGAN v2 needs anti-aliased SnakeBeta — the raw `x + sin(αx)²/β`
    // emits broadband aliases (squared-sine harmonics above Nyquist fold back as
    // click-like artifacts; verified ~2k inter-sample jumps > 30 % FS on JFK
    // prompt). AA is on by default. Set INDEXTTS_VOCODER_RAW=1 to opt out
    // (e.g. for A/B benchmarking against the broken-but-faster GPU path).
    bool use_aa = true;

    // Anti-aliased SnakeBeta params (only used when use_aa=true)
    std::vector<aa_snake_params*> aa_params;

    void clear_aa_params() {
        for (auto* p : aa_params)
            delete p;
        aa_params.clear();
    }

    ~indextts_voc_context() {
        clear_aa_params();
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        if (ctx_w) {
            ggml_free(ctx_w);
        }
        if (buf_w) {
            ggml_backend_buffer_free(buf_w);
        }
        if (backend && backend != backend_cpu) {
            ggml_backend_free(backend);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

namespace {

// ── SnakeBeta activation (native ggml ops, GPU-accelerable) ─────
//
// SnakeBeta: x + sin(α·x)² / β  where α=exp(log_alpha), β=exp(log_beta).
// Expressed as native ggml ops so the entire BigVGAN graph stays on GPU.
//
// This is the RAW path — kept for opt-in benchmarking via
// INDEXTTS_VOCODER_RAW=1. BigVGAN v2 paper wraps this activation in 2× AA
// resampling (upsample → activate → downsample, Kaiser-windowed sinc) for a
// reason: x + sin(α·x)² is non-band-limited, and on real TTS prompts the raw
// path produces ~2k sample-to-sample jumps > 30 % FS — audible as broadband
// click/buzz. AA is the default; see `aa_snake_beta` below.

static ggml_tensor* snake_beta_raw(ggml_context* ctx, ggml_tensor* x, ggml_tensor* log_alpha, ggml_tensor* log_beta) {
    if (!log_alpha || !log_beta)
        return x;
    int C = (int)log_alpha->ne[0];
    ggml_tensor* alpha = ggml_exp(ctx, ggml_reshape_2d(ctx, log_alpha, 1, C));
    ggml_tensor* beta = ggml_exp(ctx, ggml_reshape_2d(ctx, log_beta, 1, C));
    ggml_tensor* ax = ggml_mul(ctx, x, alpha);
    ggml_tensor* sin_ax = ggml_sin(ctx, ax);
    ggml_tensor* sin2 = ggml_mul(ctx, sin_ax, sin_ax);
    ggml_tensor* term = ggml_div(ctx, sin2, beta);
    return ggml_add(ctx, x, term);
}

// ── Anti-aliased SnakeBeta (CPU, default; opt out via INDEXTTS_VOCODER_RAW=1)
//
// Full Activation1d: upsample 2× → SnakeBeta → downsample 2×.
// CPU-only for now (the depthwise FIR doesn't have a ggml GPU kernel that we
// trust on Metal). Hot-loop uses thread-local pre-allocated scratch + pre-×2
// upsample filter to minimise overhead vs the raw GPU path.

static void aa_snake_beta_op(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    const auto* p = (const aa_snake_params*)userdata;
    const int T = (int)src->ne[0];
    const int C = (int)src->ne[1];
    const float* x_in = (const float*)src->data;
    float* x_out = (float*)dst->data;
    const int K = (int)p->us_filter_x2.size();
    const int up_pad = K / 2 - 1;
    const int up_pad_left = up_pad * 2 + (K - 2) / 2;
    const int up_pad_right = up_pad * 2 + (K - 2 + 1) / 2;
    const int ds_pad_left = K / 2 - 1;
    const int ds_pad_right = K / 2;

    const int c_start = (C * ith) / nth;
    const int c_end = (C * (ith + 1)) / nth;

    // Grab this worker's scratch buffers (resized on first use; capacity sticks
    // for the rest of the run since T is bounded by the largest BigVGAN layer).
    const int T_padded = T + 2 * up_pad;
    const int T_up = (T_padded - 1) * 2 + K;
    const int T_cropped = T_up - up_pad_left - up_pad_right;
    const int T_ds_padded = T_cropped + ds_pad_left + ds_pad_right;

    auto& padded = p->scratch_padded[ith];
    auto& upsampled = p->scratch_upsampled[ith];
    auto& ds_padded = p->scratch_dspadded[ith];
    auto& snake_tmp = p->scratch_snake[ith];
    // Step C-1 A/B knob — INDEXTTS_AA_SCALAR=1 forces the scalar paths for the
    // SnakeBeta and downsample stages so we can bench Accelerate's contribution.
    static const bool s_force_scalar = getenv("INDEXTTS_AA_SCALAR") != nullptr;
    if ((int)padded.size() < T_padded)
        padded.resize(T_padded);
    if ((int)upsampled.size() < T_up)
        upsampled.resize(T_up);
    if ((int)ds_padded.size() < T_ds_padded)
        ds_padded.resize(T_ds_padded);
    if ((int)snake_tmp.size() < T_cropped)
        snake_tmp.resize(T_cropped);

    // Cache the pre-scaled (×2) upsample filter and the downsample filter
    // locally so the inner loops touch dense stack memory, not p-> pointer.
    const float* uf2 = p->us_filter_x2.data();
    const float* df = p->ds_filter.data();

    for (int c = c_start; c < c_end; c++) {
        const float alpha_c = (c < p->C) ? p->alpha[c] : 1.0f;
        const float inv_beta = (c < p->C) ? (1.0f / p->beta[c]) : 1.0f;
        const float* x_in_c = x_in + (size_t)c * T;
        float* x_out_c = x_out + (size_t)c * T;

        // Edge-replication padding for upsample.
        const float left_edge = x_in_c[0];
        const float right_edge = x_in_c[T - 1];
        for (int t = 0; t < up_pad; t++)
            padded[t] = left_edge;
        std::memcpy(padded.data() + up_pad, x_in_c, (size_t)T * sizeof(float));
        for (int t = 0; t < up_pad; t++)
            padded[up_pad + T + t] = right_edge;

        // Zero-stuff upsample by 2 + FIR (Kaiser-windowed sinc, pre-×2 baked in).
        // Hot scatter loop — input sample t lands at output positions t*2..t*2+K-1
        // and accumulates into existing partials. K=12 puts 12 muladds per input;
        // the compiler unrolls cleanly so vDSP doesn't beat this here (scatter
        // doesn't map to vDSP_conv without an extra polyphase split).
        std::memset(upsampled.data(), 0, (size_t)T_up * sizeof(float));
        for (int t = 0; t < T_padded; t++) {
            const float v = padded[t];
            float* dst_row = upsampled.data() + t * 2;
            for (int k = 0; k < K; k++)
                dst_row[k] += v * uf2[k];
        }

        // SnakeBeta in-place on the cropped upsampled range.
        // Step C-1: Accelerate's vvsinf + vDSP_vsma takes this from scalar
        // (sinf + 3 muladds × T_cropped) to one block of vector sin + one
        // fused-multiply-add pass. ~2× per-call on M1.
        float* cropped = upsampled.data() + up_pad_left;
#ifdef __APPLE__
        if (!s_force_scalar) {
            float* tmp = snake_tmp.data();
            int n = T_cropped;
            vDSP_vsmul(cropped, 1, &alpha_c, tmp, 1, (vDSP_Length)n);
            vvsinf(tmp, tmp, &n); // sin in place; supports aliasing
            vDSP_vsq(tmp, 1, tmp, 1, (vDSP_Length)n);
            vDSP_vsma(tmp, 1, &inv_beta, cropped, 1, cropped, 1, (vDSP_Length)n);
        } else
#endif
        {
            for (int t = 0; t < T_cropped; t++) {
                const float v = cropped[t];
                const float s = sinf(alpha_c * v);
                cropped[t] = v + inv_beta * s * s;
            }
        }

        // Edge-replication padding for downsample.
        const float c_left = cropped[0];
        const float c_right = cropped[T_cropped - 1];
        for (int t = 0; t < ds_pad_left; t++)
            ds_padded[t] = c_left;
        std::memcpy(ds_padded.data() + ds_pad_left, cropped, (size_t)T_cropped * sizeof(float));
        for (int t = 0; t < ds_pad_right; t++)
            ds_padded[ds_pad_left + T_cropped + t] = c_right;

        // Stride-2 downsample FIR.
        // Step C-1: vDSP_desamp fuses K-tap FIR + stride-2 decimation into one
        // call backed by NEON. On M1 this is roughly 3× the scalar 12-mul loop.
        const int T_out_ds = (T_ds_padded - K) / 2 + 1;
        const int T_final = std::min(T_out_ds, T);
#ifdef __APPLE__
        if (!s_force_scalar) {
            vDSP_desamp(ds_padded.data(), /*decimation*/ 2, df, x_out_c, (vDSP_Length)T_final, (vDSP_Length)K);
        } else
#endif
        {
            for (int t = 0; t < T_final; t++) {
                const float* row = ds_padded.data() + t * 2;
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += row[k] * df[k];
                x_out_c[t] = sum;
            }
        }
        for (int t = T_final; t < T; t++)
            x_out_c[t] = 0.0f;
    }
}

static ggml_tensor* aa_snake_beta(ggml_context* ctx, ggml_tensor* x, ggml_tensor* log_alpha, ggml_tensor* log_beta,
                                  ggml_tensor* us_filter_t, ggml_tensor* ds_filter_t,
                                  std::vector<aa_snake_params*>& params_storage) {
    if (!log_alpha || !log_beta) {
        return x;
    }

    auto* p = new aa_snake_params();
    params_storage.push_back(p);

    int C = (int)log_alpha->ne[0];
    p->C = C;
    p->alpha.resize(C);
    p->beta.resize(C);

    std::vector<float> la(C), lb(C);
    ggml_backend_tensor_get(log_alpha, la.data(), 0, C * sizeof(float));
    ggml_backend_tensor_get(log_beta, lb.data(), 0, C * sizeof(float));
    for (int i = 0; i < C; i++) {
        p->alpha[i] = expf(la[i]);
        p->beta[i] = expf(lb[i]);
    }

    {
        std::vector<float> us(12, 1.0f / 12.0f);
        if (us_filter_t) {
            int flen = (int)ggml_nelements(us_filter_t);
            us.resize(flen);
            ggml_backend_tensor_get(us_filter_t, us.data(), 0, flen * sizeof(float));
        }
        // Bake the ×2 gain (from zero-stuff upsampling) into the filter so the
        // hot loop is one mul, not two.
        p->us_filter_x2.resize(us.size());
        for (size_t i = 0; i < us.size(); ++i)
            p->us_filter_x2[i] = us[i] * 2.0f;
    }
    if (ds_filter_t) {
        int flen = (int)ggml_nelements(ds_filter_t);
        p->ds_filter.resize(flen);
        ggml_backend_tensor_get(ds_filter_t, p->ds_filter.data(), 0, flen * sizeof(float));
    } else {
        p->ds_filter.resize(p->us_filter_x2.size());
        for (size_t i = 0; i < p->ds_filter.size(); ++i)
            p->ds_filter[i] = p->us_filter_x2[i] * 0.5f;
    }

    // Pre-allocate per-thread scratch slots. ggml_map_custom1 is invoked with
    // up to ggml_cpu_n_threads() workers; AA_SCRATCH_MAX_THREADS is a fixed
    // upper bound (GGML_N_TASKS_MAX is the sentinel −1, not a count). Each
    // worker writes only its own slot, so the outer vector is fixed-size and
    // race-free.
    constexpr int AA_SCRATCH_MAX_THREADS = 64;
    p->scratch_padded.resize(AA_SCRATCH_MAX_THREADS);
    p->scratch_upsampled.resize(AA_SCRATCH_MAX_THREADS);
    p->scratch_dspadded.resize(AA_SCRATCH_MAX_THREADS);
    p->scratch_snake.resize(AA_SCRATCH_MAX_THREADS);

    // Bound worker count at the scratch-slot count; ggml will further cap at
    // the actual thread pool size.
    return ggml_map_custom1(ctx, x, aa_snake_beta_op, AA_SCRATCH_MAX_THREADS, p);
}

// ── Step B (v2): native-ggml-ops AA path ────────────────────────
//
// Same semantics as `aa_snake_beta_op` but expressed entirely from native
// ggml primitives so the chain can run on the same backend as the rest of
// the BigVGAN graph (Metal, CUDA, CPU — wherever ggml supports `conv_1d`,
// `concat`, `scale`, `sin/exp/mul/add`). Gated behind
// INDEXTTS_AA_BACKEND=native; default stays on the proven CPU custom op.
//
// Two fixes vs the first attempt:
//
// 1. **Output length.** The earlier "zero-stuff + stride-1 conv1d" mismatch
//    against PyTorch's `conv_transpose1d(K=12, stride=2)` is closed by
//    using `p0 = K - 1 = 11` on the upsample `ggml_conv_1d`. Output length
//    `(2·T_p + 22 - 12)/1 + 1 = 2·T_p + 11`, which after cropping
//    `up_pad_left + (up_pad_right+1)` reduces to `2·T` — matching torch.
//
// 2. **Reshape after truncating view.** `ggml_view_3d` that narrows ne[0]
//    leaves a non-contiguous tensor; downstream `ggml_reshape_2d` then
//    silently re-strides into wrong layout. Fixed by `ggml_cont` between
//    every truncating view and the following reshape.
//
// Key trick: `ggml_conv_1d` treats `ne[2]` of its data input as a batch
// dim and applies the same kernel independently to each batch. Reshape
// [T, C] → [T, 1, C] and use the `[K, 1, 1]` filter shipped in the GGUF
// — depthwise-equivalent, zero filter blowup.
//
// All shape inputs:
//   x        : [T, C]    F32, time-fastest channel-major (ggml convention)
//   log_alpha: [C]       F32
//   log_beta : [C]       F32
//   us_filter: [K, 1, 1] F32  (12-tap Kaiser sinc, sum=1)
//   ds_filter: [K, 1, 1] F32  (12-tap Kaiser sinc, sum=1)
// Returns: [T, C] F32 — same shape as input, so the BigVGAN graph after AA
//                       sees no shape drift.
static ggml_tensor* aa_snake_beta_native(ggml_context* ctx, ggml_tensor* x, ggml_tensor* log_alpha,
                                         ggml_tensor* log_beta, ggml_tensor* us_filter, ggml_tensor* ds_filter) {
    if (!log_alpha || !log_beta || !us_filter || !ds_filter) {
        return x;
    }
    const int T = (int)x->ne[0];
    const int C = (int)x->ne[1];
    const int K = (int)us_filter->ne[0]; // 12

    // Mirror Activation1d (`alias_free_activation/torch/resample.py`).
    const int up_pad = K / 2 - 1;                          // 5
    const int up_pad_left = up_pad * 2 + (K - 2) / 2;      // 15
    const int up_pad_right = up_pad * 2 + (K - 2 + 1) / 2; // 15
    const int ds_pad_left = K / 2 - 1;                     // 5
    const int ds_pad_right = K / 2;                        // 6
    const int conv1d_pad_up = K - 1; // 11 — closes the K-1 length gap vs torch conv_transpose_1d

    // ── 1. Replicate-pad x by up_pad along ne[0] ──────────────────
    ggml_tensor* x_first = ggml_view_2d(ctx, x, 1, C, x->nb[1], 0);
    ggml_tensor* x_last = ggml_view_2d(ctx, x, 1, C, x->nb[1], (size_t)(T - 1) * x->nb[0]);

    ggml_tensor* tmpl_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, up_pad, C);
    ggml_tensor* lefts = ggml_repeat(ctx, x_first, tmpl_pad);
    ggml_tensor* rights = ggml_repeat(ctx, x_last, tmpl_pad);

    ggml_tensor* x_p = ggml_concat(ctx, lefts, x, /*dim=*/0);
    x_p = ggml_concat(ctx, x_p, rights, /*dim=*/0);
    x_p = ggml_cont(ctx, x_p); // [T_p, C]
    const int T_p = T + 2 * up_pad;

    // ── 2. Zero-stuff upsample 2× ─────────────────────────────────
    // [T_p, C] → [1, T_p, 1, C] → concat-axis-0 with zero copy → [2, T_p, 1, C]
    //   → reshape to [2·T_p, 1, C].  Putting the "stuff bit" on ne[0] makes
    //   the flatten interleave correctly.
    ggml_tensor* x_4 = ggml_reshape_4d(ctx, x_p, 1, T_p, 1, C);
    ggml_tensor* zeros4 = ggml_scale(ctx, x_4, 0.0f);
    ggml_tensor* stf4 = ggml_concat(ctx, x_4, zeros4, /*dim=*/0); // [2, T_p, 1, C]
    stf4 = ggml_cont(ctx, stf4);
    ggml_tensor* stf3 = ggml_reshape_3d(ctx, stf4, 2 * T_p, 1, C); // [2·T_p, 1, C]

    // ── 3. Upsample FIR: conv_1d K=12 stride 1 pad=K-1=11 ────────
    // Bake the ×2 zero-stuff gain into the filter.
    // Output: (2·T_p + 22 - 12)/1 + 1 = 2·T_p + 11.  After crop (up_pad_left +
    // up_pad_right + 1) → 2·T_p - 30 = 2·T (since T_p = T + 10).
    ggml_tensor* us_x2 = ggml_scale(ctx, us_filter, 2.0f);
    ggml_tensor* up_3 = ggml_conv_1d(ctx, us_x2, stf3, /*s*/ 1, /*p*/ conv1d_pad_up, /*d*/ 1);
    const int T_up_full = (int)up_3->ne[0];
    // We want exactly T_cropped = 2·T = T_up_full - up_pad_left - (up_pad_right + 1).
    const int crop_right = up_pad_right + (T_up_full - up_pad_left - up_pad_right - 2 * T);
    const int T_crop = T_up_full - up_pad_left - crop_right;
    if (T_crop != 2 * T) {
        // Sanity: shapes off → graceful fallback to raw SnakeBeta (no AA).
        return snake_beta_raw(ctx, x, log_alpha, log_beta);
    }
    up_3 = ggml_cont(ctx, up_3);
    ggml_tensor* up_crop3 =
        ggml_view_3d(ctx, up_3, T_crop, 1, C, up_3->nb[1], up_3->nb[2], (size_t)up_pad_left * up_3->nb[0]);
    up_crop3 = ggml_cont(ctx, up_crop3); // Fix #2 — cont AFTER truncating view
    ggml_tensor* up_2d = ggml_reshape_2d(ctx, up_crop3, T_crop, C);

    // ── 4. SnakeBeta (native ggml ops) ────────────────────────────
    ggml_tensor* a2d = ggml_reshape_2d(ctx, log_alpha, 1, C);
    ggml_tensor* b2d = ggml_reshape_2d(ctx, log_beta, 1, C);
    ggml_tensor* ea = ggml_exp(ctx, a2d);
    ggml_tensor* inveb = ggml_exp(ctx, ggml_neg(ctx, b2d));
    ggml_tensor* xa = ggml_mul(ctx, up_2d, ea);
    ggml_tensor* s = ggml_sin(ctx, xa);
    ggml_tensor* s2 = ggml_mul(ctx, s, s);
    ggml_tensor* term = ggml_mul(ctx, s2, inveb);
    ggml_tensor* x_sb = ggml_add(ctx, up_2d, term); // [T_crop, C]

    // ── 5. Replicate-pad for downsample ───────────────────────────
    ggml_tensor* xsb_first = ggml_view_2d(ctx, x_sb, 1, C, x_sb->nb[1], 0);
    ggml_tensor* xsb_last = ggml_view_2d(ctx, x_sb, 1, C, x_sb->nb[1], (size_t)(T_crop - 1) * x_sb->nb[0]);

    ggml_tensor* tmpl_dl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ds_pad_left, C);
    ggml_tensor* tmpl_dr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ds_pad_right, C);
    ggml_tensor* ds_l = ggml_repeat(ctx, xsb_first, tmpl_dl);
    ggml_tensor* ds_r = ggml_repeat(ctx, xsb_last, tmpl_dr);

    ggml_tensor* xsb_p = ggml_concat(ctx, ds_l, x_sb, /*dim=*/0);
    xsb_p = ggml_concat(ctx, xsb_p, ds_r, /*dim=*/0);
    xsb_p = ggml_cont(ctx, xsb_p);
    const int T_dsp = T_crop + ds_pad_left + ds_pad_right;

    // ── 6. Stride-2 downsample FIR ───────────────────────────────
    // Output: (T_dsp - K)/2 + 1 = (2·T + 5 + 6 - 12)/2 + 1 = T.
    ggml_tensor* xsb_p3 = ggml_reshape_3d(ctx, xsb_p, T_dsp, 1, C);
    ggml_tensor* down3 = ggml_conv_1d(ctx, ds_filter, xsb_p3, /*s*/ 2, /*p*/ 0, /*d*/ 1);
    const int T_out = (int)down3->ne[0];
    if (T_out != T) {
        return snake_beta_raw(ctx, x, log_alpha, log_beta);
    }
    down3 = ggml_cont(ctx, down3); // Fix #2 — cont before reshape
    return ggml_reshape_2d(ctx, down3, T, C);
}

// Dispatch by env: INDEXTTS_AA_BACKEND=native picks the ggml-native path;
// INDEXTTS_AA_BACKEND=op (or =metal) picks the new fused `ggml_aa_snake_beta`
// op (Step C-2); anything else (or unset) stays on the proven CPU custom-op
// path.
static bool aa_use_native() {
    const char* v = getenv("INDEXTTS_AA_BACKEND");
    return v && (v[0] == 'n' || v[0] == 'N');
}
static bool aa_use_opvariant() {
    const char* v = getenv("INDEXTTS_AA_BACKEND");
    // Match "op", "Op", "metal", "Metal".
    return v && (v[0] == 'o' || v[0] == 'O' || v[0] == 'm' || v[0] == 'M');
}

// ── ECAPA-TDNN speaker encoder ──────────────────────────────────
//
// Architecture mirrors the qwen3_tts ECAPA-TDNN (SE-Res2Net blocks,
// ASP pooling, final FC → 512d). The key difference is that IndexTTS
// ships BatchNorm tensors unfused, so we apply BN explicitly:
//   y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
//
// Tensor naming in the GGUF:
//   se.b.0.c.{weight,bias}           — initial TDNN conv
//   se.b.0.n.{weight,bias,rm,rv}     — BatchNorm
//   se.b.{1,2,3}.tdnn{1,2}.*         — SE-Res2Net TDNNs
//   se.b.{1,2,3}.r2n.b.{0-6}.*       — Res2Net internal
//   se.b.{1,2,3}.se_block.conv{1,2}.*— SE squeeze/excite
//   se.mfa.*                          — multi-frame aggregation
//   se.asp.*                          — attentive stat pooling
//   se.asp_bn.*                       — ASP BatchNorm
//   se.fc.*                           — final linear

// BatchNorm: y = gamma * (x - rm) / sqrt(rv + eps) + beta
// x is [C, T] channels-first. gamma/beta/rm/rv are [C].
//
// Since rv + eps is done in graph-compute context (which is no-alloc),
// we use ggml_clamp to floor rv at eps rather than adding a scalar.
// This is safe because rv (running_variance) is always >= 0 and we just
// need to avoid division by zero.
static ggml_tensor* ecapa_bn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma, ggml_tensor* beta, ggml_tensor* rm,
                             ggml_tensor* rv) {
    if (!gamma || !beta || !rm || !rv) {
        return x;
    }
    const int C = (int)x->ne[0];
    ggml_tensor* mean = ggml_reshape_2d(ctx, rm, C, 1);
    ggml_tensor* var = ggml_reshape_2d(ctx, rv, C, 1);
    ggml_tensor* g = ggml_reshape_2d(ctx, gamma, C, 1);
    ggml_tensor* b = ggml_reshape_2d(ctx, beta, C, 1);

    x = ggml_sub(ctx, x, mean);
    // rv + eps: clamp rv at eps to avoid sqrt(0), then add eps via scale trick:
    // sqrt(rv + eps) ≈ sqrt(max(rv, eps)) for rv >= 0.
    // More precisely: use ggml_clamp to ensure minimum of eps.
    ggml_tensor* var_safe = ggml_clamp(ctx, var, 1e-5f, 1e30f);
    ggml_tensor* denom = ggml_sqrt(ctx, var_safe);
    x = ggml_div(ctx, x, denom);
    x = ggml_add(ctx, ggml_mul(ctx, x, g), b);
    return x;
}

// Conv1d with reflect padding + BatchNorm + ReLU.
// Input/output: [C, T] channels-first.
static ggml_tensor* ecapa_tdnn_bn_relu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cw, ggml_tensor* cb,
                                       ggml_tensor* nw, ggml_tensor* nb, ggml_tensor* nrm, ggml_tensor* nrv,
                                       int dilation) {
    const int K = (int)cw->ne[0];
    const int pad = (K - 1) * dilation / 2;
    // [C, T] → [T, C] for ggml_pad_reflect_1d / ggml_conv_1d
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    if (pad > 0) {
        x = ggml_pad_reflect_1d(ctx, x, pad, pad);
    }
    x = ggml_conv_1d(ctx, cw, x, 1, 0, dilation);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // back to [C, T]
    if (cb) {
        x = ggml_add(ctx, x, cb);
    }
    x = ecapa_bn(ctx, x, nw, nb, nrm, nrv);
    x = ggml_relu(ctx, x);
    return x;
}

// SE block: global mean pool → linear → ReLU → linear → sigmoid → scale
static ggml_tensor* ecapa_se(ggml_context* ctx, ggml_tensor* x, ggml_tensor* c1w, ggml_tensor* c1b, ggml_tensor* c2w,
                             ggml_tensor* c2b) {
    const int T_se = (int)x->ne[1];
    // Global mean: [C, T] → mean → [C, 1]
    ggml_tensor* m = ggml_cont(
        ctx,
        ggml_transpose(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, x))), 1.0f / T_se)));
    auto w1 = ggml_reshape_2d(ctx, c1w, c1w->ne[1], c1w->ne[2]);
    ggml_tensor* h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, m), c1b));
    auto w2 = ggml_reshape_2d(ctx, c2w, c2w->ne[1], c2w->ne[2]);
    ggml_tensor* sc = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h), c2b));
    return ggml_mul(ctx, x, ggml_repeat(ctx, sc, x));
}

// Res2Net block: split into 8 chunks, pass through 7 TDNNs with dilation.
static ggml_tensor* ecapa_res2net(ggml_context* ctx, ggml_tensor* x, const std::map<std::string, ggml_tensor*>& ts,
                                  const std::string& prefix, int dilation) {
    const int T_r2n = (int)x->ne[1];
    const int chunk = 64; // 512 / 8
    ggml_tensor* outs[8];
    for (int i = 0; i < 8; i++) {
        ggml_tensor* ci =
            ggml_cont(ctx, ggml_view_2d(ctx, x, chunk, T_r2n, x->nb[1], (size_t)i * chunk * sizeof(float)));
        if (i == 0) {
            outs[i] = ci;
            continue;
        }
        ggml_tensor* in = (i == 1) ? ci : ggml_add(ctx, ci, outs[i - 1]);
        // TDNN with BN for Res2Net internal block (i-1)
        char key[128];
        std::snprintf(key, sizeof(key), "%s.%d.c.weight", prefix.c_str(), i - 1);
        ggml_tensor* cw = T(ts, key);
        std::snprintf(key, sizeof(key), "%s.%d.c.bias", prefix.c_str(), i - 1);
        ggml_tensor* cb = T(ts, key);
        std::snprintf(key, sizeof(key), "%s.%d.n.weight", prefix.c_str(), i - 1);
        ggml_tensor* nw = T(ts, key);
        std::snprintf(key, sizeof(key), "%s.%d.n.bias", prefix.c_str(), i - 1);
        ggml_tensor* nb = T(ts, key);
        std::snprintf(key, sizeof(key), "%s.%d.n.rm", prefix.c_str(), i - 1);
        ggml_tensor* nrm = T(ts, key);
        std::snprintf(key, sizeof(key), "%s.%d.n.rv", prefix.c_str(), i - 1);
        ggml_tensor* nrv = T(ts, key);
        outs[i] = ecapa_tdnn_bn_relu(ctx, in, cw, cb, nw, nb, nrm, nrv, dilation);
    }
    ggml_tensor* out = outs[0];
    for (int i = 1; i < 8; i++) {
        out = ggml_concat(ctx, out, outs[i], 0);
    }
    return out;
}

// SE-Res2Net block: TDNN1 → Res2Net → TDNN2 → SE → residual
static ggml_tensor* ecapa_se_res2net(ggml_context* ctx, ggml_tensor* x, const std::map<std::string, ggml_tensor*>& ts,
                                     int blk_idx, int dilation) {
    ggml_tensor* res = x;
    char prefix[64];

    // TDNN1
    std::snprintf(prefix, sizeof(prefix), "se.b.%d.tdnn1", blk_idx);
    x = ecapa_tdnn_bn_relu(ctx, x, T(ts, std::string(prefix) + ".c.weight"), T(ts, std::string(prefix) + ".c.bias"),
                           T(ts, std::string(prefix) + ".n.weight"), T(ts, std::string(prefix) + ".n.bias"),
                           T(ts, std::string(prefix) + ".n.rm"), T(ts, std::string(prefix) + ".n.rv"), 1);

    // Res2Net
    std::snprintf(prefix, sizeof(prefix), "se.b.%d.r2n.b", blk_idx);
    x = ecapa_res2net(ctx, x, ts, prefix, dilation);

    // TDNN2
    std::snprintf(prefix, sizeof(prefix), "se.b.%d.tdnn2", blk_idx);
    x = ecapa_tdnn_bn_relu(ctx, x, T(ts, std::string(prefix) + ".c.weight"), T(ts, std::string(prefix) + ".c.bias"),
                           T(ts, std::string(prefix) + ".n.weight"), T(ts, std::string(prefix) + ".n.bias"),
                           T(ts, std::string(prefix) + ".n.rm"), T(ts, std::string(prefix) + ".n.rv"), 1);

    // SE
    std::snprintf(prefix, sizeof(prefix), "se.b.%d.se_block", blk_idx);
    x = ecapa_se(ctx, x, T(ts, std::string(prefix) + ".conv1.conv.weight"),
                 T(ts, std::string(prefix) + ".conv1.conv.bias"), T(ts, std::string(prefix) + ".conv2.conv.weight"),
                 T(ts, std::string(prefix) + ".conv2.conv.bias"));

    return ggml_add(ctx, x, res);
}

// ASP: Attentive Statistics Pooling → [2*C, 1]
static ggml_tensor* ecapa_asp(ggml_context* ctx, ggml_tensor* x, const std::map<std::string, ggml_tensor*>& ts) {
    const int T_asp = (int)x->ne[1];
    // Global statistics
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* m1C = ggml_scale(ctx, ggml_sum_rows(ctx, xT), 1.0f / T_asp);
    ggml_tensor* mC1 = ggml_cont(ctx, ggml_transpose(ctx, m1C));
    ggml_tensor* mCT = ggml_repeat(ctx, mC1, x);
    ggml_tensor* d2 = ggml_mul(ctx, ggml_sub(ctx, x, mCT), ggml_sub(ctx, x, mCT));
    ggml_tensor* s1C =
        ggml_sqrt(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, d2))), 1.0f / T_asp));
    ggml_tensor* sCT = ggml_repeat(ctx, ggml_cont(ctx, ggml_transpose(ctx, s1C)), x);

    // [x, mean, std] → TDNN (with BN + ReLU) → tanh → conv → softmax
    ggml_tensor* att = ggml_concat(ctx, ggml_concat(ctx, x, mCT, 0), sCT, 0);
    att = ecapa_tdnn_bn_relu(ctx, att, T(ts, "se.asp.tdnn.c.weight"), T(ts, "se.asp.tdnn.c.bias"),
                             T(ts, "se.asp.tdnn.n.weight"), T(ts, "se.asp.tdnn.n.bias"), T(ts, "se.asp.tdnn.n.rm"),
                             T(ts, "se.asp.tdnn.n.rv"), 1);
    att = ggml_tanh(ctx, att);

    ggml_tensor* asp_cw = T(ts, "se.asp.c.weight");
    ggml_tensor* asp_cb = T(ts, "se.asp.c.bias");
    auto cw2d = ggml_reshape_2d(ctx, asp_cw, asp_cw->ne[1], asp_cw->ne[2]);
    att = ggml_add(ctx, ggml_mul_mat(ctx, cw2d, att), asp_cb);
    att = ggml_cont(ctx, ggml_transpose(ctx, att));
    att = ggml_soft_max(ctx, att);
    att = ggml_cont(ctx, ggml_transpose(ctx, att));

    // Weighted mean and std → [2C, 1]
    ggml_tensor* wx = ggml_mul(ctx, att, x);
    ggml_tensor* wm = ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, wx)))));
    ggml_tensor* wmCT = ggml_repeat(ctx, wm, x);
    ggml_tensor* dd = ggml_sub(ctx, x, wmCT);
    ggml_tensor* ws = ggml_sqrt(
        ctx,
        ggml_cont(ctx,
                  ggml_transpose(
                      ctx, ggml_sum_rows(
                               ctx, ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, att, ggml_mul(ctx, dd, dd))))))));
    return ggml_concat(ctx, wm, ws, 0); // [2C, 1]
}

// Build the full ECAPA-TDNN graph. Input: [n_mels=100, T] mel spectrogram.
// Output: [512] speaker embedding.
static ggml_cgraph* build_ecapa_graph(indextts_voc_context* c, int T_mel) {
    auto& ts = c->tensors;
    const size_t n_nodes = 8192;
    // Use persistent compute_meta buffer (same as BigVGAN graph) so the graph
    // remains valid after this function returns.  A local vector would be freed
    // on return, leaving dangling pointers in the returned graph.
    const size_t need = ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false);
    if (c->compute_meta.size() < need) {
        c->compute_meta.resize(need);
    }
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_nodes, false);

    // Input: [100, T]
    ggml_tensor* h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 100, T_mel);
    ggml_set_name(h, "ecapa_mel");
    ggml_set_input(h);

    // Block 0: initial TDNN (100→512, k=5)
    h = ecapa_tdnn_bn_relu(ctx0, h, T(ts, "se.b.0.c.weight"), T(ts, "se.b.0.c.bias"), T(ts, "se.b.0.n.weight"),
                           T(ts, "se.b.0.n.bias"), T(ts, "se.b.0.n.rm"), T(ts, "se.b.0.n.rv"), 1);

    // 3 SE-Res2Net blocks with dilations 2, 3, 4
    static const int dilations[3] = {2, 3, 4};
    ggml_tensor* blk_outs[3];
    for (int i = 0; i < 3; i++) {
        h = ecapa_se_res2net(ctx0, h, ts, i + 1, dilations[i]);
        blk_outs[i] = h;
    }

    // MFA: concatenate outputs of 3 blocks → TDNN (1536→1536, k=1)
    ggml_tensor* mfa_in = ggml_concat(ctx0, ggml_concat(ctx0, blk_outs[0], blk_outs[1], 0), blk_outs[2], 0);
    h = ecapa_tdnn_bn_relu(ctx0, mfa_in, T(ts, "se.mfa.c.weight"), T(ts, "se.mfa.c.bias"), T(ts, "se.mfa.n.weight"),
                           T(ts, "se.mfa.n.bias"), T(ts, "se.mfa.n.rm"), T(ts, "se.mfa.n.rv"), 1);

    // ASP: attentive stat pooling → [3072, 1]
    h = ecapa_asp(ctx0, h, ts);

    // ASP BatchNorm
    h = ecapa_bn(ctx0, h, T(ts, "se.asp_bn.norm.weight"), T(ts, "se.asp_bn.norm.bias"), T(ts, "se.asp_bn.norm.rm"),
                 T(ts, "se.asp_bn.norm.rv"));

    // Final FC: [3072] → [512]
    ggml_tensor* fcw = T(ts, "se.fc.conv.weight");
    ggml_tensor* fcb = T(ts, "se.fc.conv.bias");
    auto fc2d = ggml_reshape_2d(ctx0, fcw, fcw->ne[1], fcw->ne[2]);
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, fc2d, h), fcb);
    h = ggml_reshape_1d(ctx0, h, 512);

    ggml_set_name(h, "spk_emb_out");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);
    ggml_free(ctx0);
    return gf;
}

// Compute 100-band mel spectrogram for ECAPA-TDNN.
// Input: mono float32 PCM at 24kHz.
// Output: (T, 100) row-major float32 mel, T written to *T_out.
static std::vector<float> compute_ecapa_mel(const float* pcm, int n_samples, int* T_out) {
    const int n_fft = 1024, hop = 256, n_mels = 100, sr = 24000;
    const float fmin = 0.0f, fmax = 12000.0f;

    // Input is already 24kHz (resampled by the backend caller).
    // No further resampling needed — the ECAPA-TDNN in IndexTTS's BigVGAN
    // vocoder operates at 24kHz natively.

    const int pad = (n_fft - hop) / 2; // 384

    // Reflect-pad audio
    std::vector<float> audio_p(n_samples + 2 * pad, 0.0f);
    for (int i = 0; i < pad; i++) {
        audio_p[i] = pcm[std::min(pad - i, n_samples - 1)];
    }
    for (int i = 0; i < n_samples; i++) {
        audio_p[pad + i] = pcm[i];
    }
    for (int i = 0; i < pad; i++) {
        audio_p[pad + n_samples + i] = pcm[std::max(n_samples - 2 - i, 0)];
    }

    // Periodic Hann window
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; i++) {
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)n_fft));
    }

    const int n_freqs = n_fft / 2 + 1;
    auto mel_fb = core_mel::build_htk_fb(sr, n_fft, n_mels, fmin, fmax);

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = 1e-7f;
    p.spec_kind = core_mel::SpecKind::Magnitude;
    p.norm = core_mel::Normalization::None;
    p.layout = core_mel::Layout::TimeMels;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.center_pad = false; // already reflect-padded

    int T = 0;
    auto mel = core_mel::compute(audio_p.data(), (int)audio_p.size(), hann.data(), n_fft, mel_fb.data(), n_freqs,
                                 core_fft::fft_radix2_wrapper, p, T);
    if (T_out) {
        *T_out = T;
    }
    return mel;
}

// Run ECAPA-TDNN: PCM → 512d speaker embedding.
static std::vector<float> run_ecapa_tdnn(indextts_voc_context* c, const float* ref_pcm, int ref_n_samples) {
    // Check if ECAPA weights exist
    if (!T(c->tensors, "se.b.0.c.weight")) {
        if (c->verbosity >= 1) {
            fprintf(stderr, "indextts-voc: ECAPA-TDNN weights not found, using zero speaker embedding\n");
        }
        return std::vector<float>(512, 0.0f);
    }

    // Compute mel
    int T_mel = 0;
    auto mel = compute_ecapa_mel(ref_pcm, ref_n_samples, &T_mel);
    if (mel.empty() || T_mel <= 0) {
        fprintf(stderr, "indextts-voc: failed to compute mel for ECAPA\n");
        return std::vector<float>(512, 0.0f);
    }

    // Limit mel to 500 frames (~5s reference) — longer reference doesn't improve quality
    if (T_mel > 500) {
        T_mel = 500;
        mel.resize((size_t)T_mel * 100);
    }

    if (c->verbosity >= 1) {
        fprintf(stderr, "indextts-voc: ECAPA mel: %d frames x 100 bands\n", T_mel);
    }

    // Convert mel (T, 100) row-major → ggml [C=100, T] flat layout
    std::vector<float> mel_CT((size_t)100 * T_mel);
    for (int t = 0; t < T_mel; t++) {
        for (int ch = 0; ch < 100; ch++) {
            mel_CT[(size_t)ch + (size_t)t * 100] = mel[(size_t)t * 100 + ch];
        }
    }

    // Build and run graph
    ggml_cgraph* gf = build_ecapa_graph(c, T_mel);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "indextts-voc: failed to alloc ECAPA graph\n");
        return std::vector<float>(512, 0.0f);
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ecapa_mel"), mel_CT.data(), 0, mel_CT.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "indextts-voc: ECAPA compute failed\n");
        return std::vector<float>(512, 0.0f);
    }

    std::vector<float> emb(512);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "spk_emb_out"), emb.data(), 0, 512 * sizeof(float));

    if (c->verbosity >= 1) {
        float norm = 0.0f;
        for (float v : emb) {
            norm += v * v;
        }
        fprintf(stderr, "indextts-voc: ECAPA speaker embedding norm = %.4f\n", sqrtf(norm));
    }

    return emb;
}

// ── Build BigVGAN graph ─────────────────────────────────────────

static ggml_cgraph* build_bigvgan_graph(indextts_voc_context* c, int T_in) {
    const auto& hp = c->hp;
    auto& ts = c->tensors;

    const size_t n_nodes = 32768;
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_nodes, false);

    // Input: latent from GPT has ggml shape ne[0]=D, ne[1]=T (each position is D contiguous floats).
    // Conv1d in ggml expects ne[0]=T (time in innermost). So we create the tensor matching
    // the GPT layout and then transpose it for conv operations.
    ggml_tensor* x_raw = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.gpt_dim, T_in);
    ggml_set_name(x_raw, "latent_input");
    ggml_set_input(x_raw);
    // Transpose to ne[0]=T_in, ne[1]=gpt_dim for conv1d
    ggml_tensor* x = ggml_cont(ctx0, ggml_transpose(ctx0, x_raw));

    // Speaker embedding input: (1, spk_emb_dim)
    ggml_tensor* spk = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, hp.spk_emb_dim);
    ggml_set_name(spk, "spk_emb");
    ggml_set_input(spk);

    // conv_pre: Conv1d(gpt_dim, upsample_initial_ch, k=7, pad=3)
    ggml_tensor* cpre_w = T(ts, "conv_pre.weight");
    ggml_tensor* cpre_b = T(ts, "conv_pre.bias");
    if (cpre_w) {
        x = ggml_conv_1d(ctx0, cpre_w, x, 1, 3, 1);
        if (cpre_b) {
            x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, cpre_b, 1, (int)cpre_b->ne[0]));
        }
        ggml_set_name(x, "dbg_conv_pre");
        ggml_set_output(x);
    }

    // Speaker conditioning: x += cond_layer(spk_emb)
    // cond_layer: Conv1d(512, 1536, k=1)
    // spk is (1, 512), cond_layer.weight is (1, 512, 1536)
    // After conv1d: (1, 1536) -> broadcast add to x (T, 1536)
    {
        ggml_tensor* cond_w = T(ts, "cond_layer.weight");
        ggml_tensor* cond_b = T(ts, "cond_layer.bias");
        if (cond_w) {
            ggml_tensor* cond = ggml_conv_1d(ctx0, cond_w, spk, 1, 0, 1);
            if (cond_b) {
                cond = ggml_add(ctx0, cond, ggml_reshape_2d(ctx0, cond_b, 1, (int)cond_b->ne[0]));
            }
            x = ggml_add(ctx0, x, cond);
        }
    }

    // Channel sizes after each upsample: 1536 -> 768 -> 384 -> 192 -> 96 -> 48 -> 24
    int ch = hp.upsample_initial_ch;

    for (int i = 0; i < hp.num_upsamples; i++) {
        int s = hp.upsample_rates[i];
        int k = hp.upsample_kernel_sizes[i];
        int ch_out = ch / 2;

        // SnakeBeta activation before upsample
        {
            char aname[64], bname[64];
            // The pre-upsample snake uses the first activation of the first
            // resblock at this level. Actually, looking at BigVGAN source:
            // ups[i] has its own activation. Let me check tensor names...
            // In BigVGAN Python: self.ups has LeakyReLU before each
            // ConvTranspose1d. But with snake_logscale, it uses SnakeBeta.
            // The Python code stores it as ups[i] = [Activation, ConvTranspose1d].
            // In the GGUF, ups.{i}.0 is the ConvTranspose1d weight.
            // The activation before ups is actually just the first activation
            // of the first resblock at this level... No, let me look more carefully.
            //
            // Actually, BigVGAN v2 uses the pattern:
            //   for i in range(num_upsamples):
            //       x = act(x)      # SnakeBeta (part of ups module)
            //       x = ups[i](x)   # ConvTranspose1d
            //
            // But looking at our GGUF, there's no separate activation tensor
            // for the ups modules. The activations in the GGUF are all under
            // resblocks.{n}.act.{m}. So the pre-upsample activation
            // must be handled differently.
            //
            // Looking at the BigVGAN code more carefully:
            //   self.ups = nn.ModuleList()
            //   for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            //       self.ups.append(nn.ModuleList([
            //           Activation1d(activation=SnakeBeta(...)),
            //           ...ConvTranspose1d...
            //       ]))
            //
            // IndexTTS BigVGAN has NO activation before upsample — self.ups[i]
            // contains only ConvTranspose1d. The SnakeBeta activations are all
            // inside the AMPBlock1 resblocks.
            (void)aname;
            (void)bname;
        }

        // ConvTranspose1d upsample
        {
            char wn[32], bn[32];
            std::snprintf(wn, sizeof(wn), "ups.%d.0.weight", i);
            std::snprintf(bn, sizeof(bn), "ups.%d.0.bias", i);
            ggml_tensor* up_w = T(ts, wn);
            ggml_tensor* up_b = T(ts, bn);
            if (up_w) {
                int T_cur = (int)x->ne[0];
                x = ggml_conv_transpose_1d(ctx0, up_w, x, s, 0, 1);
                // Crop padding: output with p=0 is (T-1)*s+k, we want T*s
                int p = (k - s) / 2;
                if (p > 0) {
                    int T_want = T_cur * s;
                    int C_out_t = (int)x->ne[1];
                    x = ggml_view_2d(ctx0, x, T_want, C_out_t, x->nb[1], (size_t)p * x->nb[0]);
                    x = ggml_cont(ctx0, x);
                }
                if (up_b) {
                    x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, up_b, 1, (int)up_b->ne[0]));
                }
            }
        }

        // Per-level speaker conditioning: x += conds[i](spk_emb)
        {
            char wn[32], bn[32];
            std::snprintf(wn, sizeof(wn), "conds.%d.weight", i);
            std::snprintf(bn, sizeof(bn), "conds.%d.bias", i);
            ggml_tensor* cw = T(ts, wn);
            ggml_tensor* cb = T(ts, bn);
            if (cw) {
                ggml_tensor* cond = ggml_conv_1d(ctx0, cw, spk, 1, 0, 1);
                if (cb) {
                    cond = ggml_add(ctx0, cond, ggml_reshape_2d(ctx0, cb, 1, (int)cb->ne[0]));
                }
                x = ggml_add(ctx0, x, cond);
            }
        }

        // 3 ResBlocks in parallel, averaged
        ggml_tensor* rb_sum = nullptr;
        ggml_tensor* rb_input = x;

        for (int j = 0; j < hp.num_kernels; j++) {
            x = rb_input; // reset to same input
            int rb_idx = i * hp.num_kernels + j;
            int rb_k = hp.resblock_kernel_sizes[j];

            ggml_tensor* rb_residual = x;

            // 3 dilated passes per resblock
            for (int d = 0; d < 3; d++) {
                int dil = hp.resblock_dilations[d];
                int act_idx_1 = d * 2;     // activation indices: 0,2,4
                int act_idx_2 = d * 2 + 1; // activation indices: 1,3,5

                char key[80];

                // SnakeBeta activation 1
                std::snprintf(key, sizeof(key), "resb.%d.act.%d.act.alpha", rb_idx, act_idx_1);
                ggml_tensor* alpha1 = T(ts, key);
                std::snprintf(key, sizeof(key), "resb.%d.act.%d.act.beta", rb_idx, act_idx_1);
                ggml_tensor* beta1 = T(ts, key);
                if (c->use_aa) {
                    std::snprintf(key, sizeof(key), "resb.%d.act.%d.us.filter", rb_idx, act_idx_1);
                    ggml_tensor* usf1 = T(ts, key);
                    std::snprintf(key, sizeof(key), "resb.%d.act.%d.ds.filter", rb_idx, act_idx_1);
                    ggml_tensor* dsf1 = T(ts, key);
                    if (aa_use_native()) {
                        x = aa_snake_beta_native(ctx0, x, alpha1, beta1, usf1, dsf1);
                    } else if (aa_use_opvariant()) {
                        x = ggml_aa_snake_beta(ctx0, x, alpha1, beta1, usf1, dsf1);
                    } else {
                        x = aa_snake_beta(ctx0, x, alpha1, beta1, usf1, dsf1, c->aa_params);
                    }
                } else {
                    x = snake_beta_raw(ctx0, x, alpha1, beta1);
                }

                // Conv1d with dilation
                int pad1 = (rb_k * dil - dil) / 2;
                std::snprintf(key, sizeof(key), "resb.%d.convs1.%d.weight", rb_idx, d);
                ggml_tensor* c1w = T(ts, key);
                std::snprintf(key, sizeof(key), "resb.%d.convs1.%d.bias", rb_idx, d);
                ggml_tensor* c1b = T(ts, key);
                if (c1w) {
                    x = ggml_conv_1d(ctx0, c1w, x, 1, pad1, dil);
                    if (c1b) {
                        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c1b, 1, (int)c1b->ne[0]));
                    }
                }

                // SnakeBeta activation 2
                std::snprintf(key, sizeof(key), "resb.%d.act.%d.act.alpha", rb_idx, act_idx_2);
                ggml_tensor* alpha2 = T(ts, key);
                std::snprintf(key, sizeof(key), "resb.%d.act.%d.act.beta", rb_idx, act_idx_2);
                ggml_tensor* beta2 = T(ts, key);
                if (c->use_aa) {
                    std::snprintf(key, sizeof(key), "resb.%d.act.%d.us.filter", rb_idx, act_idx_2);
                    ggml_tensor* usf2 = T(ts, key);
                    std::snprintf(key, sizeof(key), "resb.%d.act.%d.ds.filter", rb_idx, act_idx_2);
                    ggml_tensor* dsf2 = T(ts, key);
                    if (aa_use_native()) {
                        x = aa_snake_beta_native(ctx0, x, alpha2, beta2, usf2, dsf2);
                    } else if (aa_use_opvariant()) {
                        x = ggml_aa_snake_beta(ctx0, x, alpha2, beta2, usf2, dsf2);
                    } else {
                        x = aa_snake_beta(ctx0, x, alpha2, beta2, usf2, dsf2, c->aa_params);
                    }
                } else {
                    x = snake_beta_raw(ctx0, x, alpha2, beta2);
                }

                // Conv2d (dilation=1)
                int pad2 = (rb_k - 1) / 2;
                std::snprintf(key, sizeof(key), "resb.%d.convs2.%d.weight", rb_idx, d);
                ggml_tensor* c2w = T(ts, key);
                std::snprintf(key, sizeof(key), "resb.%d.convs2.%d.bias", rb_idx, d);
                ggml_tensor* c2b = T(ts, key);
                if (c2w) {
                    x = ggml_conv_1d(ctx0, c2w, x, 1, pad2, 1);
                    if (c2b) {
                        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c2b, 1, (int)c2b->ne[0]));
                    }
                }

                // Residual connection
                x = ggml_add(ctx0, x, rb_residual);
                rb_residual = x;
            }

            // Accumulate for averaging
            if (!rb_sum) {
                rb_sum = x;
            } else {
                rb_sum = ggml_add(ctx0, rb_sum, x);
            }
        }
        // Average the 3 ResBlock outputs
        x = ggml_scale(ctx0, rb_sum, 1.0f / 3.0f);

        ch = ch_out;
    }

    // Final SnakeBeta activation
    {
        ggml_tensor* alpha_post = T(ts, "activation_post.act.alpha");
        ggml_tensor* beta_post = T(ts, "activation_post.act.beta");
        if (c->use_aa) {
            ggml_tensor* usf_post = T(ts, "activation_post.us.filter");
            ggml_tensor* dsf_post = T(ts, "activation_post.ds.filter");
            if (aa_use_native()) {
                x = aa_snake_beta_native(ctx0, x, alpha_post, beta_post, usf_post, dsf_post);
            } else if (aa_use_opvariant()) {
                x = ggml_aa_snake_beta(ctx0, x, alpha_post, beta_post, usf_post, dsf_post);
            } else {
                x = aa_snake_beta(ctx0, x, alpha_post, beta_post, usf_post, dsf_post, c->aa_params);
            }
        } else {
            x = snake_beta_raw(ctx0, x, alpha_post, beta_post);
        }
    }

    // conv_post: Conv1d(24, 1, k=7, pad=3)
    {
        ggml_tensor* cpost_w = T(ts, "conv_post.weight");
        ggml_tensor* cpost_b = T(ts, "conv_post.bias");
        if (cpost_w) {
            x = ggml_conv_1d(ctx0, cpost_w, x, 1, 3, 1);
            if (cpost_b) {
                x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, cpost_b, 1, (int)cpost_b->ne[0]));
            }
        }
    }

    // tanh output clamp
    x = ggml_tanh(ctx0, x);

    ggml_set_name(x, "audio_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);
    return gf;
}

} // namespace

// ── Public C ABI ────────────────────────────────────────────────

extern "C" struct indextts_voc_context* indextts_voc_init(const char* path, int n_threads, bool use_gpu) {
    if (!path) {
        return nullptr;
    }

    auto* c = new indextts_voc_context();
    c->n_threads = n_threads > 0 ? n_threads : 4;

    // BigVGAN v2 anti-aliased SnakeBeta is on by default — the raw activation
    // produces audible aliasing (verified). Two env knobs:
    //   INDEXTTS_VOCODER_RAW=1 → force raw SnakeBeta (legacy path; aliased)
    //   INDEXTTS_VOCODER_AA=0  → same, alternate spelling
    //   INDEXTTS_VOCODER_AA=1  → force AA (also the default)
    const char* raw_env = getenv("INDEXTTS_VOCODER_RAW");
    const char* aa_env = getenv("INDEXTTS_VOCODER_AA");
    if (raw_env && raw_env[0] == '1') {
        c->use_aa = false;
    } else if (aa_env && aa_env[0] == '0') {
        c->use_aa = false;
    } else {
        c->use_aa = true;
    }

    // Pass 1: metadata
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            delete c;
            return nullptr;
        }
        auto& hp = c->hp;
        hp.gpt_dim = (int)core_gguf::kv_u32(g, "indextts.bigvgan.gpt_dim", (uint32_t)hp.gpt_dim);
        hp.upsample_initial_ch =
            (int)core_gguf::kv_u32(g, "indextts.bigvgan.upsample_initial_channel", (uint32_t)hp.upsample_initial_ch);
        hp.num_upsamples = (int)core_gguf::kv_u32(g, "indextts.bigvgan.num_upsamples", (uint32_t)hp.num_upsamples);
        hp.num_kernels = (int)core_gguf::kv_u32(g, "indextts.bigvgan.num_kernels", (uint32_t)hp.num_kernels);
        hp.spk_emb_dim = (int)core_gguf::kv_u32(g, "indextts.bigvgan.speaker_embedding_dim", (uint32_t)hp.spk_emb_dim);
        hp.sampling_rate = (int)core_gguf::kv_u32(g, "indextts.sampling_rate", (uint32_t)hp.sampling_rate);
        hp.hop_size = (int)core_gguf::kv_u32(g, "indextts.bigvgan.hop_size", (uint32_t)hp.hop_size);

        // Read array hyperparameters from GGUF KV
        // upsample_rates: [4, 4, 4, 4, 2, 2]
        // upsample_kernel_sizes: [8, 8, 4, 4, 4, 4]
        // resblock_kernel_sizes: [3, 7, 11]
        // resblock_dilation_sizes: [1, 3, 5, 1, 3, 5, 1, 3, 5] (flattened 3x3)
        // These are stored as GGUF arrays. We read them via gguf_find_key + raw data.
        {
            int key_id = gguf_find_key(g, "indextts.bigvgan.upsample_rates");
            if (key_id >= 0) {
                const int n = std::min((int)gguf_get_arr_n(g, key_id), 6);
                for (int ii = 0; ii < n; ii++) {
                    hp.upsample_rates[ii] = (int)((const uint32_t*)gguf_get_arr_data(g, key_id))[ii];
                }
            }
        }
        {
            int key_id = gguf_find_key(g, "indextts.bigvgan.upsample_kernel_sizes");
            if (key_id >= 0) {
                const int n = std::min((int)gguf_get_arr_n(g, key_id), 6);
                for (int ii = 0; ii < n; ii++) {
                    hp.upsample_kernel_sizes[ii] = (int)((const uint32_t*)gguf_get_arr_data(g, key_id))[ii];
                }
            }
        }
        {
            int key_id = gguf_find_key(g, "indextts.bigvgan.resblock_kernel_sizes");
            if (key_id >= 0) {
                const int n = std::min((int)gguf_get_arr_n(g, key_id), 3);
                for (int ii = 0; ii < n; ii++) {
                    hp.resblock_kernel_sizes[ii] = (int)((const uint32_t*)gguf_get_arr_data(g, key_id))[ii];
                }
            }
        }
        {
            int key_id = gguf_find_key(g, "indextts.bigvgan.resblock_dilation_sizes");
            if (key_id >= 0) {
                const int n = std::min((int)gguf_get_arr_n(g, key_id), 9);
                // Stored flattened: 3 groups of 3. We just take the first 3
                // (dilations are the same for all resblock groups).
                for (int ii = 0; ii < std::min(n, 3); ii++) {
                    hp.resblock_dilations[ii] = (int)((const uint32_t*)gguf_get_arr_data(g, key_id))[ii];
                }
            }
        }

        core_gguf::free_metadata(g);

        fprintf(stderr, "indextts-voc: BigVGAN gpt_dim=%d init_ch=%d ups=%d kernels=%d spk=%d sr=%d hop=%d\n",
                hp.gpt_dim, hp.upsample_initial_ch, hp.num_upsamples, hp.num_kernels, hp.spk_emb_dim, hp.sampling_rate,
                hp.hop_size);
        fprintf(stderr, "indextts-voc: upsample_rates=[%d,%d,%d,%d,%d,%d] kernels=[%d,%d,%d,%d,%d,%d]\n",
                hp.upsample_rates[0], hp.upsample_rates[1], hp.upsample_rates[2], hp.upsample_rates[3],
                hp.upsample_rates[4], hp.upsample_rates[5], hp.upsample_kernel_sizes[0], hp.upsample_kernel_sizes[1],
                hp.upsample_kernel_sizes[2], hp.upsample_kernel_sizes[3], hp.upsample_kernel_sizes[4],
                hp.upsample_kernel_sizes[5]);
        fprintf(stderr, "indextts-voc: resblock_kernels=[%d,%d,%d] dilations=[%d,%d,%d]\n", hp.resblock_kernel_sizes[0],
                hp.resblock_kernel_sizes[1], hp.resblock_kernel_sizes[2], hp.resblock_dilations[0],
                hp.resblock_dilations[1], hp.resblock_dilations[2]);
    }

    // Backend — Step A: when AA SnakeBeta is on, the activation is a CPU-only
    // custom op (`ggml_map_custom1`). Mixing it with a GPU backend forces a
    // Metal → CPU → Metal sync per AMP block (≈ 20 sites). The sync overhead
    // exceeds whatever Metal wins on the matmuls around it, so the GPU+AA
    // combo measures ≈ 25 % SLOWER than CPU+AA on M1. Auto-fall to CPU here so
    // users don't have to remember `--no-gpu` for IndexTTS. Set
    // INDEXTTS_VOC_FORCE_GPU=1 to opt back into the slow mixed path (useful
    // for the benchmark history once Step B/C lift the AA op to GPU).
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "indextts-voc: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    const bool force_gpu_with_aa = getenv("INDEXTTS_VOC_FORCE_GPU") && getenv("INDEXTTS_VOC_FORCE_GPU")[0] == '1';
    // Native-ops AA (Step B) and the new ggml_aa_snake_beta op (Step C-2) both
    // run on whichever backend owns the graph — no Metal↔CPU sync at each AA
    // site. The auto-CPU fallback only applies to the legacy map_custom1 path.
    const bool aa_blocks_gpu = c->use_aa && !aa_use_native() && !aa_use_opvariant() && !force_gpu_with_aa;
    const bool effective_use_gpu = use_gpu && !aa_blocks_gpu;
    c->backend = effective_use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        c->backend = c->backend_cpu;
    }
    if (use_gpu && aa_blocks_gpu) {
        fprintf(stderr, "indextts-voc: GPU disabled for vocoder — AA SnakeBeta custom op is CPU-only;\n");
        fprintf(stderr, "             Metal↔CPU sync per AMP block makes GPU slower. Override knobs:\n");
        fprintf(stderr, "             INDEXTTS_VOC_FORCE_GPU=1 (keep mixed-backend custom op on GPU),\n");
        fprintf(stderr, "             or INDEXTTS_VOCODER_RAW=1 (aliased fully-GPU; audibly clicks).\n");
    }

    // Pass 2: weights
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path, c->backend, "indextts-voc", wl)) {
            delete c;
            return nullptr;
        }
        c->ctx_w = wl.ctx;
        c->buf_w = wl.buf;
        c->tensors = std::move(wl.tensors);
    }

    // Verify critical tensors exist
    if (!T(c->tensors, "conv_pre.weight") || !T(c->tensors, "conv_post.weight")) {
        fprintf(stderr, "indextts-voc: missing conv_pre or conv_post weights\n");
        delete c;
        return nullptr;
    }

    // Compute scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = c->backend;
        if (c->backend != c->backend_cpu) {
            backends[n_be++] = c->backend_cpu;
        }
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false));
    }

    fprintf(stderr, "indextts-voc: loaded %zu tensors from '%s'\n", c->tensors.size(), path);
    fprintf(stderr, "indextts-voc: SnakeBeta = %s\n",
            c->use_aa ? "anti-aliased (CPU; default)" : "raw (aliased, faster; INDEXTTS_VOCODER_RAW=1)");
    return c;
}

extern "C" float* indextts_voc_generate(struct indextts_voc_context* ctx, const float* latent, int T_in,
                                        const float* spk_emb, int* out_n) {
    if (!ctx || !latent || T_in <= 0 || !out_n) {
        return nullptr;
    }
    *out_n = 0;

    const auto& hp = ctx->hp;

    // Compute expected output length
    int T_audio = T_in;
    for (int i = 0; i < hp.num_upsamples; i++) {
        T_audio *= hp.upsample_rates[i];
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "indextts-voc: generating audio T_in=%d -> T_audio=%d (%.2f sec)\n", T_in, T_audio,
                (float)T_audio / hp.sampling_rate);
    }

    // Build graph
    ggml_cgraph* gf = build_bigvgan_graph(ctx, T_in);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "indextts-voc: failed to alloc BigVGAN graph\n");
        return nullptr;
    }

    // Set latent input: GPT produces [n_latent, D] row-major (each position is D contiguous floats).
    // The tensor was created as (D, T_in) to match this layout.
    {
        ggml_tensor* inp = ggml_graph_get_tensor(gf, "latent_input");
        if (!inp) {
            fprintf(stderr, "indextts-voc: latent_input tensor not found in graph\n");
            return nullptr;
        }
        ggml_backend_tensor_set(inp, latent, 0, (size_t)T_in * hp.gpt_dim * sizeof(float));
    }

    // Set speaker embedding: shape (1, spk_emb_dim)
    {
        ggml_tensor* spk_t = ggml_graph_get_tensor(gf, "spk_emb");
        if (spk_t) {
            std::vector<float> spk_data(hp.spk_emb_dim, 0.0f);
            if (spk_emb) {
                std::memcpy(spk_data.data(), spk_emb, hp.spk_emb_dim * sizeof(float));
            }
            ggml_backend_tensor_set(spk_t, spk_data.data(), 0, hp.spk_emb_dim * sizeof(float));
        }
    }

    // Compute
    auto t0 = std::chrono::high_resolution_clock::now();
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "indextts-voc: BigVGAN compute failed\n");
        ctx->clear_aa_params();
        return nullptr;
    }
    ctx->clear_aa_params();
    auto t1 = std::chrono::high_resolution_clock::now();
    const bool bench = getenv("INDEXTTS_BENCH") != nullptr;
    if (ctx->verbosity >= 1 || bench) {
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const char* mode = ctx->use_aa ? (ctx->backend == ctx->backend_cpu ? "AA/CPU" : "AA/mixed") : "raw/GPU";
        fprintf(stderr, "indextts-voc: BigVGAN compute %.1f ms (%s)\n", ms, mode);
    }

    // Debug: read conv_pre output
    if (ctx->verbosity >= 1) {
        ggml_tensor* dbg = ggml_graph_get_tensor(gf, "dbg_conv_pre");
        if (dbg) {
            int n = (int)ggml_nelements(dbg);
            std::vector<float> d(n);
            ggml_backend_tensor_get(dbg, d.data(), 0, n * sizeof(float));
            float s2 = 0;
            for (float v : d)
                s2 += v * v;
            fprintf(stderr, "indextts-voc: conv_pre rms=%.4f ne=(%lld,%lld) first5=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                    sqrtf(s2 / n), (long long)dbg->ne[0], (long long)dbg->ne[1], d[0], d[1], d[2], d[3], d[4]);
        }
    }

    // Read output
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "audio_out");
    if (!out_t) {
        fprintf(stderr, "indextts-voc: audio_out tensor not found\n");
        return nullptr;
    }

    int n_samples = (int)ggml_nelements(out_t);
    float* result = (float*)malloc((size_t)n_samples * sizeof(float));
    ggml_backend_tensor_get(out_t, result, 0, (size_t)n_samples * sizeof(float));

    *out_n = n_samples;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "indextts-voc: generated %d samples (%.2f sec @ %d Hz)\n", n_samples,
                (float)n_samples / hp.sampling_rate, hp.sampling_rate);
    }

    return result;
}

extern "C" float* indextts_voc_speaker_embed(struct indextts_voc_context* ctx, const float* ref_pcm, int n_samples) {
    if (!ctx || !ref_pcm || n_samples <= 0) {
        return nullptr;
    }
    auto emb = run_ecapa_tdnn(ctx, ref_pcm, n_samples);
    if (emb.empty()) {
        return nullptr;
    }
    float* result = (float*)malloc(emb.size() * sizeof(float));
    std::memcpy(result, emb.data(), emb.size() * sizeof(float));
    return result;
}

extern "C" void indextts_voc_free(struct indextts_voc_context* ctx) {
    delete ctx;
}
