// crepe.cpp — CREPE F0 estimation runtime.
//
// This graph is a direct transcription of tools/crepe_numpy_parity.py, which
// is validated at cos=1.0 against torchcrepe on both capacities. If you change
// the graph, change that script first and re-run it.
//
// Two geometry decisions carried over from the blueprint (docs/music-
// transcription/PLAN.md):
//
//   * Layer order is pad -> conv -> RELU -> batchnorm -> maxpool. The relu is
//     BEFORE the BN, so BN cannot fold into the conv; the converter ships it
//     as a per-channel affine (scale, offset) and we apply it here.
//   * conv2..6 want an ASYMMETRIC (31, 32) pad. Metal rejects an asymmetric
//     GGML_OP_PAD, so we convolve with a symmetric p=32 and drop output
//     column 0: left-pad 32 starts each window one sample earlier than
//     left-pad 31, so our output j is the symmetric output j+1.
//
// PERFORMANCE / CORRECTNESS NOTE — why this is a persistent single-frame graph
// rather than a batched one:
//
// ggml_conv_1d nominally carries a batch dim (b = [T, IC, N]), but its N > 1
// path is broken in this ggml: it reshapes the mul_mat result to (OL, OC, N)
// from a buffer whose actual layout only coincides with that at N == 1. Batch=1
// reproduces the reference exactly (cos=1.0 through all six layers); batch=2
// already scrambles frame 0. Measured, not assumed — see docs/music-
// transcription/PLAN.md.
//
// So we run one frame per compute, and instead amortise the part that actually
// dominated: the per-frame graph BUILD + gallocr alloc. The graph is built once
// at init and reused, dispatching tensor_set -> compute -> tensor_get per frame
// (the documented persistent-graph pattern). Every input is re-set on every
// compute, because gallocr may hand an input's buffer to a later intermediate.

#include "crepe.h"

#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/crispasr_env.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

// Per-layer geometry, from torchcrepe/model.py. pad_l != pad_r for conv2..6.
struct layer_geom {
    const char* name;
    int kernel;
    int stride;
    int pad_l;
    int pad_r;
};

constexpr layer_geom kLayers[6] = {
    {"conv1", 512, 4, 254, 254}, {"conv2", 64, 1, 31, 32}, {"conv3", 64, 1, 31, 32},
    {"conv4", 64, 1, 31, 32},    {"conv5", 64, 1, 31, 32}, {"conv6", 64, 1, 31, 32},
};

constexpr int kNumLayers = 6;

// Frames per graph dispatch. Each layer becomes one GEMM over all of them,
// which is what lets BLAS/Metal reach useful throughput on a model that costs
// ~2.8 GFLOP per frame. Tail batches are zero-padded and the extra outputs
// discarded, so a single persistent graph covers every call.
constexpr int kBatch = 64;

struct crepe_layer {
    ggml_tensor* w = nullptr;         // (K, IC, OC)
    ggml_tensor* b = nullptr;         // (OC)
    ggml_tensor* bn_scale = nullptr;  // (OC)
    ggml_tensor* bn_offset = nullptr; // (OC)
};

} // namespace

struct crepe_context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 4;

    std::string capacity;
    int pitch_bins = CREPE_PITCH_BINS;
    int window_size = CREPE_WINDOW_SIZE;
    int sample_rate = CREPE_SAMPLE_RATE;
    int in_features = 0;
    float cents_per_bin = 20.0f;
    float cents_offset = 1997.3794084376191f;

    crepe_layer layers[kNumLayers];
    ggml_tensor* cls_w = nullptr;
    ggml_tensor* cls_b = nullptr;

    // Persistent single-frame graph, built once in crepe_init.
    ggml_context* g_ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor* g_in = nullptr;
    ggml_tensor* g_out = nullptr;

    // F32-baked conv kernels (see crepe_init).
    ggml_context* bake_ctx = nullptr;
    ggml_backend_buffer_t bake_buf = nullptr;

    bool debug = false;
};

namespace {

// conv1d over a batch, returning CHANNEL-FASTEST (OC, OL, N).
//
// Deliberately NOT ggml_conv_1d, for two reasons:
//
//  1. Correctness: its batch path reshapes the mul_mat result to (OL, OC, N)
//     from a buffer that is actually (OC, OL*N) — those only coincide at
//     N == 1. Batch=1 matched the reference exactly; batch=2 scrambled frame 0.
//
//  2. Cost: ggml_conv_1d ends with a permute back to (OL, OC, N), which
//     materializes the whole activation every layer. We keep the mul_mat's
//     native (OC, OL, N) instead and do all the elementwise work there, where
//     a plain (OC) vector broadcasts along ne[0] — ggml's fast path — instead
//     of the stride-0 (1, OC, 1) broadcast the other layout forces.
//
// The one transpose per layer that im2col does require is deferred until AFTER
// the 2x maxpool, so it moves half as much data.
ggml_tensor* crepe_conv1d_cf(ggml_context* c0, ggml_tensor* w, ggml_tensor* x, int stride, int pad) {
    ggml_tensor* col = ggml_im2col(c0, w, x, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32); // (IC*K, OL, N)
    ggml_tensor* w2 = ggml_reshape_2d(c0, w, w->ne[0] * w->ne[1], w->ne[2]);                 // (K*IC, OC)
    return ggml_mul_mat(c0, w2, col);                                                        // (OC, OL, N)
}

// Build the forward graph for `batch` frames. Returns the output tensor
// (pitch_bins, batch); `frames_in` receives the (window, 1, batch) input.
ggml_tensor* crepe_build_graph(crepe_context* ctx, ggml_context* c0, ggml_cgraph* gf, int batch,
                               ggml_tensor** frames_in) {
    ggml_tensor* x = ggml_new_tensor_3d(c0, GGML_TYPE_F32, ctx->window_size, 1, batch);
    ggml_set_name(x, "frames");
    ggml_set_input(x);
    *frames_in = x;

    for (int i = 0; i < kNumLayers; i++) {
        const layer_geom& g = kLayers[i];
        const crepe_layer& L = ctx->layers[i];
        const int64_t t_in = x->ne[0];

        // Symmetric pad = max(pad_l, pad_r); crop below when they differ.
        const int p = std::max(g.pad_l, g.pad_r);
        x = crepe_conv1d_cf(c0, L.w, x, g.stride, p); // (OC, OL, N) channel-fastest

        if (g.pad_l != g.pad_r) {
            // Asymmetric (31, 32) emulated as symmetric 32 -> drop OL column 0.
            // Symmetric p=32 yields OL = t_in + 1; we want columns [1, t_in].
            x = ggml_cont(c0, ggml_view_3d(c0, x, x->ne[0], t_in, x->ne[2], x->nb[1], x->nb[2], x->nb[1]));
        }

        // Elementwise, all in channel-fastest layout: a plain (OC) vector
        // broadcasts along ne[0]. ggml_conv_1d has no bias, so add it here.
        x = ggml_add(c0, x, L.b);
        x = ggml_relu(c0, x); // relu BEFORE the batchnorm — see header
        x = ggml_add(c0, ggml_mul(c0, x, L.bn_scale), L.bn_offset);

        // Pool along OL (= ne[1] here), leaving channels and batch alone.
        x = ggml_pool_2d(c0, x, GGML_OP_POOL_MAX, 1, 2, 1, 2, 0, 0); // (OC, OL/2, N)

        // im2col needs (T, IC, N) for the next layer. Transposing after the
        // pool halves the bytes moved. The last layer skips it entirely: its
        // (OC, OL, N) IS the channel-fastest flatten the classifier wants.
        if (i + 1 < kNumLayers)
            x = ggml_cont(c0, ggml_permute(c0, x, 1, 0, 2, 3)); // (OL, OC, N)
    }

    // torch does permute(0, 2, 1, 3).reshape(batch, in_features) — i.e. flatten
    // time-major with CHANNEL fastest. Our (OC, OL, N) is already exactly that,
    // so this reshape is free (no permute, no copy).
    x = ggml_reshape_2d(c0, x, x->ne[0] * x->ne[1], batch); // (in_features, N)

    x = ggml_mul_mat(c0, ctx->cls_w, x); // (pitch_bins, N)
    x = ggml_add(c0, x, ctx->cls_b);
    x = ggml_sigmoid(c0, x);

    ggml_set_name(x, "activation");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return x;
}

// Slice `n` frames starting at frame `f0` into `dst` (window * n floats),
// normalizing each frame the way torchcrepe's preprocess() does. `pcm` is the
// caller's audio; framing is virtual so we never materialize the padded copy.
void crepe_fill_frames(const float* pcm, int n_samples, int window, int hop, int f0, int n, float* dst) {
    const int half = window / 2;
    for (int i = 0; i < n; i++) {
        float* out = dst + (size_t)i * window;
        const int start = (f0 + i) * hop - half; // may be negative (zero pad)
        for (int j = 0; j < window; j++) {
            const int s = start + j;
            out[j] = (s >= 0 && s < n_samples) ? pcm[s] : 0.0f;
        }
        double mean = 0.0;
        for (int j = 0; j < window; j++)
            mean += out[j];
        mean /= window;
        double var = 0.0;
        for (int j = 0; j < window; j++) {
            out[j] -= (float)mean;
            var += (double)out[j] * out[j];
        }
        // torchcrepe: frames /= max(1e-10, std). std is the unbiased (n-1) one,
        // matching torch.std's default correction=1.
        const double sd = std::sqrt(var / std::max(1, window - 1));
        const float inv = 1.0f / (float)std::max(sd, 1e-10);
        for (int j = 0; j < window; j++)
            out[j] *= inv;
    }
}

// Weighted average of cents over +/- 4 bins around the argmax — the original
// CREPE "local average" decoder. (torchcrepe defaults to Viterbi instead; we
// expose the raw activation so a caller can do that if it wants.)
void crepe_decode_local_average(const float* act, int bins, float cents_per_bin, float cents_offset, float* f0_hz,
                                float* voiced) {
    int best = 0;
    for (int i = 1; i < bins; i++)
        if (act[i] > act[best])
            best = i;

    const int lo = std::max(0, best - 4);
    const int hi = std::min(bins - 1, best + 4);
    double num = 0.0, den = 0.0;
    for (int i = lo; i <= hi; i++) {
        const double w = act[i];
        num += w * (cents_per_bin * i + cents_offset);
        den += w;
    }
    const double cents = den > 0.0 ? num / den : (cents_per_bin * best + cents_offset);
    *f0_hz = (float)(10.0 * std::pow(2.0, cents / 1200.0));
    *voiced = act[best];
}

// Run the model over all frames, writing activations. Shared by the F0 and
// raw-activation entry points.
int crepe_run(crepe_context* ctx, const float* pcm, int n_samples, float hop_ms, int max_frames,
              std::vector<float>& activations) {
    if (!ctx || !pcm || n_samples <= 0 || max_frames <= 0)
        return 0;
    const float ms = hop_ms > 0 ? hop_ms : 10.0f;
    const int hop = (int)std::lround(ctx->sample_rate * ms / 1000.0);
    if (hop <= 0)
        return 0;
    const int n_frames = std::min(max_frames, 1 + n_samples / hop);
    if (n_frames <= 0)
        return 0;

    if (!ctx->gf || !ctx->galloc || !ctx->g_in || !ctx->g_out)
        return 0;

    activations.assign((size_t)n_frames * ctx->pitch_bins, 0.0f);
    std::vector<float> frames((size_t)kBatch * ctx->window_size);
    std::vector<float> out_buf((size_t)kBatch * ctx->pitch_bins);

    for (int f0 = 0; f0 < n_frames; f0 += kBatch) {
        const int n = std::min(kBatch, n_frames - f0);
        // Tail batch: zero the unused frames rather than reshaping the graph,
        // so one persistent graph serves every call. Their outputs are dropped.
        if (n < kBatch)
            std::fill(frames.begin() + (size_t)n * ctx->window_size, frames.end(), 0.0f);
        crepe_fill_frames(pcm, n_samples, ctx->window_size, hop, f0, n, frames.data());

        // Re-set on EVERY compute: gallocr may reuse an input's buffer as
        // scratch for a later node once that input's last use has passed.
        ggml_backend_tensor_set(ctx->g_in, frames.data(), 0, (size_t)kBatch * ctx->window_size * sizeof(float));
        if (ggml_backend_graph_compute(ctx->backend, ctx->gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "crepe: graph compute failed at frame %d\n", f0);
            return 0;
        }
        ggml_backend_tensor_get(ctx->g_out, out_buf.data(), 0, (size_t)kBatch * ctx->pitch_bins * sizeof(float));
        std::memcpy(activations.data() + (size_t)f0 * ctx->pitch_bins, out_buf.data(),
                    (size_t)n * ctx->pitch_bins * sizeof(float));
    }
    return n_frames;
}

} // namespace

extern "C" struct crepe_context* crepe_init(const char* model_path, int n_threads) {
    if (!model_path)
        return nullptr;

    auto* ctx = new crepe_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    ctx->debug = crispasr_env::get("CRISPASR_CREPE_DEBUG") != nullptr;

    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        fprintf(stderr, "crepe: cannot open %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    ctx->capacity = core_gguf::kv_str(gctx, "crepe.capacity", "full");
    ctx->pitch_bins = (int)core_gguf::kv_u32(gctx, "crepe.pitch_bins", CREPE_PITCH_BINS);
    ctx->window_size = (int)core_gguf::kv_u32(gctx, "crepe.window_size", CREPE_WINDOW_SIZE);
    ctx->sample_rate = (int)core_gguf::kv_u32(gctx, "crepe.sample_rate", CREPE_SAMPLE_RATE);
    ctx->in_features = (int)core_gguf::kv_u32(gctx, "crepe.in_features", 0);
    ctx->cents_per_bin = core_gguf::kv_f32(gctx, "crepe.cents_per_bin", 20.0f);
    ctx->cents_offset = core_gguf::kv_f32(gctx, "crepe.cents_offset", 1997.3794084376191f);
    core_gguf::free_metadata(gctx);

    // CUDA > Metal > Vulkan > CPU. CRISPASR_CREPE_NO_GPU=1 forces CPU for A/B.
    const bool no_gpu = crispasr_env::get("CRISPASR_CREPE_NO_GPU") != nullptr;
    ctx->backend = no_gpu ? nullptr : crispasr_init_gpu_backend();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!core_gguf::load_weights(model_path, ctx->backend, "crepe", ctx->wl)) {
        crepe_free(ctx);
        return nullptr;
    }

    for (int i = 0; i < kNumLayers; i++) {
        const std::string n = kLayers[i].name;
        ctx->layers[i].w = core_gguf::require(ctx->wl.tensors, (n + ".weight").c_str(), "crepe");
        ctx->layers[i].b = core_gguf::require(ctx->wl.tensors, (n + ".bias").c_str(), "crepe");
        ctx->layers[i].bn_scale = core_gguf::require(ctx->wl.tensors, (n + "_BN.scale").c_str(), "crepe");
        ctx->layers[i].bn_offset = core_gguf::require(ctx->wl.tensors, (n + "_BN.offset").c_str(), "crepe");
        if (!ctx->layers[i].w || !ctx->layers[i].b || !ctx->layers[i].bn_scale || !ctx->layers[i].bn_offset) {
            crepe_free(ctx);
            return nullptr;
        }
    }
    ctx->cls_w = core_gguf::require(ctx->wl.tensors, "classifier.weight", "crepe");
    ctx->cls_b = core_gguf::require(ctx->wl.tensors, "classifier.bias", "crepe");
    if (!ctx->cls_w || !ctx->cls_b) {
        crepe_free(ctx);
        return nullptr;
    }

    // Bake F32 copies of the conv kernels once at load.
    //
    // ggml_conv_1d casts an F16 kernel to F32 *inside the graph* when the
    // activations are F32. In a persistent graph that cast node re-runs on
    // every compute — i.e. it re-casts the whole 44 MB weight set once per
    // 10 ms frame, which measured RTF 31 on M1. Converting once here makes the
    // cast a no-op. (Same fix as qwen3-tts CODEC_FASTCONV.) Gate:
    // CRISPASR_CREPE_NO_BAKE_F32=1 restores the in-graph cast for A/B.
    if (crispasr_env::get("CRISPASR_CREPE_NO_BAKE_F32") == nullptr) {
        size_t n_f16 = 0;
        for (int i = 0; i < kNumLayers; i++)
            if (ctx->layers[i].w->type == GGML_TYPE_F16)
                n_f16++;
        if (n_f16 > 0) {
            ggml_init_params bp = {n_f16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
            ctx->bake_ctx = ggml_init(bp);
            if (!ctx->bake_ctx) {
                crepe_free(ctx);
                return nullptr;
            }
            std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pending; // dst(f32), src(f16)
            for (int i = 0; i < kNumLayers; i++) {
                ggml_tensor* src = ctx->layers[i].w;
                if (src->type != GGML_TYPE_F16)
                    continue;
                ggml_tensor* dst = ggml_new_tensor_3d(ctx->bake_ctx, GGML_TYPE_F32, src->ne[0], src->ne[1], src->ne[2]);
                pending.emplace_back(dst, src);
            }
            ctx->bake_buf = ggml_backend_alloc_ctx_tensors(ctx->bake_ctx, ctx->backend);
            if (!ctx->bake_buf) {
                crepe_free(ctx);
                return nullptr;
            }
            std::vector<ggml_fp16_t> src_h;
            std::vector<float> dst_h;
            size_t k = 0;
            for (int i = 0; i < kNumLayers; i++) {
                if (ctx->layers[i].w->type != GGML_TYPE_F16)
                    continue;
                ggml_tensor* src = pending[k].second;
                ggml_tensor* dst = pending[k].first;
                const size_t n = (size_t)ggml_nelements(src);
                src_h.resize(n);
                dst_h.resize(n);
                ggml_backend_tensor_get(src, src_h.data(), 0, n * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(src_h.data(), dst_h.data(), (int64_t)n);
                ggml_backend_tensor_set(dst, dst_h.data(), 0, n * sizeof(float));
                ctx->layers[i].w = dst;
                k++;
            }
            if (ctx->debug)
                fprintf(stderr, "crepe: baked %zu conv kernels to F32\n", n_f16);
        }
    }

    // Build the persistent single-frame graph once, and allocate it once.
    {
        const size_t nodes = 512;
        ggml_init_params ip = {nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr,
                               true};
        ctx->g_ctx = ggml_init(ip);
        if (!ctx->g_ctx) {
            crepe_free(ctx);
            return nullptr;
        }
        ctx->gf = ggml_new_graph_custom(ctx->g_ctx, nodes, false);
        ctx->g_out = crepe_build_graph(ctx, ctx->g_ctx, ctx->gf, kBatch, &ctx->g_in);

        ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ctx->galloc || !ggml_gallocr_alloc_graph(ctx->galloc, ctx->gf)) {
            fprintf(stderr, "crepe: failed to allocate compute graph\n");
            crepe_free(ctx);
            return nullptr;
        }
    }

    if (ctx->debug)
        fprintf(stderr, "crepe: capacity=%s bins=%d window=%d in_features=%d threads=%d\n", ctx->capacity.c_str(),
                ctx->pitch_bins, ctx->window_size, ctx->in_features, ctx->n_threads);
    return ctx;
}

extern "C" void crepe_free(struct crepe_context* ctx) {
    if (!ctx)
        return;
    if (ctx->galloc)
        ggml_gallocr_free(ctx->galloc);
    if (ctx->g_ctx)
        ggml_free(ctx->g_ctx);
    if (ctx->bake_buf)
        ggml_backend_buffer_free(ctx->bake_buf);
    if (ctx->bake_ctx)
        ggml_free(ctx->bake_ctx);
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" int crepe_n_frames(const struct crepe_context* ctx, int n_samples, float hop_ms) {
    if (!ctx || n_samples <= 0)
        return 0;
    const float ms = hop_ms > 0 ? hop_ms : 10.0f;
    const int hop = (int)std::lround(ctx->sample_rate * ms / 1000.0);
    if (hop <= 0)
        return 0;
    return 1 + n_samples / hop;
}

extern "C" int crepe_compute_f0(struct crepe_context* ctx, const float* pcm_16k, int n_samples, float hop_ms,
                                struct crepe_frame* out, int max_frames) {
    if (!out)
        return 0;
    std::vector<float> act;
    const int n = crepe_run(ctx, pcm_16k, n_samples, hop_ms, max_frames, act);
    if (n <= 0)
        return 0;
    const float ms = hop_ms > 0 ? hop_ms : 10.0f;
    for (int i = 0; i < n; i++) {
        out[i].time_ms = i * ms;
        crepe_decode_local_average(act.data() + (size_t)i * ctx->pitch_bins, ctx->pitch_bins, ctx->cents_per_bin,
                                   ctx->cents_offset, &out[i].f0_hz, &out[i].voiced_prob);
    }
    return n;
}

extern "C" int crepe_compute_activation(struct crepe_context* ctx, const float* pcm_16k, int n_samples, float hop_ms,
                                        float* out, int max_frames) {
    if (!out)
        return 0;
    std::vector<float> act;
    const int n = crepe_run(ctx, pcm_16k, n_samples, hop_ms, max_frames, act);
    if (n <= 0)
        return 0;
    std::memcpy(out, act.data(), act.size() * sizeof(float));
    return n;
}

extern "C" const char* crepe_capacity(const struct crepe_context* ctx) {
    return ctx ? ctx->capacity.c_str() : "";
}
