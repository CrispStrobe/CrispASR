// pyannote_seg.cpp — native runtime for pyannote-segmentation-3.0.
//
// Architecture: SincNet front-end → 3× MaxPool → 4× bidirectional LSTM
//               → 3× Linear → LogSoftmax.
// The model is tiny (6 MB) and runs entirely on CPU in manual F32.

#include "pyannote_seg.h"
#include "core/gguf_loader.h"
#if defined(CRISPASR_USE_ONNXRUNTIME)
#include "core/ort_session.h"
#endif
#include "core/crispasr_env.h"
#include "core/powerset.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `PYANNOTE_SEG_BENCH=1` for per-stage timings.
// ===========================================================================

static bool pyannote_seg_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_PYANNOTE_SEG_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct pyannote_seg_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit pyannote_seg_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~pyannote_seg_bench_stage() {
        if (!pyannote_seg_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  pyannote_seg_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Model
// ===========================================================================

struct pyannote_lstm {
    // ONNX convention: W (2, 4H, C_in), R (2, 4H, H), B (2, 8H)
    // Direction 0 = forward, 1 = backward.
    ggml_tensor* W = nullptr; // weight_ih: (2, 4*hidden, input_size)
    ggml_tensor* R = nullptr; // weight_hh: (2, 4*hidden, hidden_size)
    ggml_tensor* B = nullptr; // bias:      (2, 8*hidden)
    int input_size = 0;
    int hidden_size = 128;
};

struct pyannote_model {
    // SincNet front-end
    ggml_tensor* sinc_filters = nullptr;                // (80, 1, 251) learned sinc
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr; // (60, 80, 5)
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr; // (60, 60, 5)
    ggml_tensor *norm0_w = nullptr, *norm0_b = nullptr; // InstanceNorm (80)
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr; // (60)
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr; // (60)

    // 4 bidirectional LSTM layers
    pyannote_lstm lstm[4];

    // Linear classifiers
    ggml_tensor* linear0_b = nullptr; // (128,)
    ggml_tensor* linear1_b = nullptr; // (128,)
    ggml_tensor* linear2_b = nullptr; // (7,) — actually the last classifier's bias
    ggml_tensor* matmul0_w = nullptr; // (256, 128)
    ggml_tensor* matmul1_w = nullptr; // (128, 128)
    ggml_tensor* matmul2_w = nullptr; // (128, 7)

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_t backend = nullptr; // owns buf — must outlive it on Metal
    std::map<std::string, ggml_tensor*> tensors;
};

struct pyannote_seg_context {
    pyannote_model model;
    int n_threads = 4;
#if defined(CRISPASR_USE_ONNXRUNTIME)
    Ort::Session* ort_session = nullptr; // onnxruntime path (path ends in ".onnx")
#endif
};

// ===========================================================================
// Loader
// ===========================================================================

static bool pyannote_load(pyannote_model& m, const char* path) {
    // pyannote_seg_run() is a manual CPU F32 implementation that directly
    // dereferences tensor->data, so weights must live in host memory.
    ggml_backend_t backend = core_cpu_backend::init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "pyannote_seg", wl)) {
        ggml_backend_free(backend);
        return false;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    m.backend = backend; // keep alive until pyannote_seg_free
    m.tensors = std::move(wl.tensors);

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = m.tensors.find(name);
        return it != m.tensors.end() ? it->second : nullptr;
    };

    m.sinc_filters = get("pyannote..sincnet.conv1d.0.Concat_2_output_0");
    m.conv1_w = get("pyannote.sincnet.conv1d.1.weight");
    m.conv1_b = get("pyannote.sincnet.conv1d.1.bias");
    m.conv2_w = get("pyannote.sincnet.conv1d.2.weight");
    m.conv2_b = get("pyannote.sincnet.conv1d.2.bias");
    m.norm0_w = get("pyannote.sincnet.norm1d.0.weight");
    m.norm0_b = get("pyannote.sincnet.norm1d.0.bias");
    m.norm1_w = get("pyannote.sincnet.norm1d.1.weight");
    m.norm1_b = get("pyannote.sincnet.norm1d.1.bias");
    m.norm2_w = get("pyannote.sincnet.norm1d.2.weight");
    m.norm2_b = get("pyannote.sincnet.norm1d.2.bias");

    // LSTM layers
    int lstm_input_sizes[] = {60, 256, 256, 256};
    const char* lstm_names[] = {
        "pyannote.onnx::LSTM_783", "pyannote.onnx::LSTM_784", "pyannote.onnx::LSTM_785", "pyannote.onnx::LSTM_826",
        "pyannote.onnx::LSTM_827", "pyannote.onnx::LSTM_828", "pyannote.onnx::LSTM_869", "pyannote.onnx::LSTM_870",
        "pyannote.onnx::LSTM_871", "pyannote.onnx::LSTM_912", "pyannote.onnx::LSTM_913", "pyannote.onnx::LSTM_914",
    };
    for (int i = 0; i < 4; i++) {
        m.lstm[i].B = get(lstm_names[i * 3 + 0]);
        m.lstm[i].W = get(lstm_names[i * 3 + 1]);
        m.lstm[i].R = get(lstm_names[i * 3 + 2]);
        m.lstm[i].input_size = lstm_input_sizes[i];
        m.lstm[i].hidden_size = 128;
    }

    m.linear0_b = get("pyannote.linear.0.bias");
    m.linear1_b = get("pyannote.linear.1.bias");
    m.linear2_b = get("pyannote.ortshared_1_1_7_0_token_109"); // (7,) last bias
    m.matmul0_w = get("pyannote.onnx::MatMul_915");            // (256, 128)
    m.matmul1_w = get("pyannote.onnx::MatMul_916");            // (128, 128)
    m.matmul2_w = get("pyannote.onnx::MatMul_917");            // (128, 7)

    if (!m.sinc_filters || !m.lstm[0].W || !m.matmul0_w) {
        fprintf(stderr, "pyannote_seg: missing critical tensors\n");
        return false;
    }
    return true;
}

// ===========================================================================
// ggml forward (default) — threaded SIMD convs/matmuls on the CPU backend.
// Only the inherently sequential LSTM recurrence stays as plain loops
// (contiguous dot products, one thread per direction); the per-timestep
// input projections W@x — 2/3 of the LSTM FLOPs — are batched into one
// mul_mat over the whole sequence per layer/direction.
// CRISPASR_PYANNOTE_LEGACY=1 selects the original scalar path (A/B ground
// truth; the ggml path is validated frame-for-frame against it).
// ===========================================================================

static bool pyannote_use_legacy() {
    static const bool v = [] {
        const char* e = std::getenv("CRISPASR_PYANNOTE_LEGACY");
        return e && *e && *e != '0';
    }();
    return v;
}

// PYANNOTE_SEG_DUMP=<path>: write the (T,7) log-prob output as raw F32 for
// A/B comparison between the ggml and legacy paths.
static void pyannote_seg_dump(const float* logp, int T) {
    const char* p = crispasr_env::get("CRISPASR_PYANNOTE_SEG_DUMP");
    if (!p || !logp)
        return;
    if (FILE* f = fopen(p, "wb")) {
        fwrite(logp, sizeof(float), (size_t)T * 7, f);
        fclose(f);
    }
}

// Run a small CPU-backend graph: alloc, set inputs, compute. Caller builds
// the graph in `ctx0` and passes the inputs to set.
struct pyannote_graph_run {
    ggml_context* ctx0 = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t ga = nullptr;

    explicit pyannote_graph_run(size_t n_nodes) {
        ggml_init_params ip = {n_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_nodes, false), nullptr,
                               true};
        ctx0 = ggml_init(ip);
        if (ctx0)
            gf = ggml_new_graph_custom(ctx0, n_nodes, false);
    }
    ~pyannote_graph_run() {
        if (ga)
            ggml_gallocr_free(ga);
        if (ctx0)
            ggml_free(ctx0);
    }
    bool compute(ggml_backend_t backend, ggml_tensor* out) {
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(ga, gf))
            return false;
        return true; // caller sets inputs, then calls ggml_backend_graph_compute
    }
};

// SincNet front-end: conv(1→80,k=251,s=10) → |·| → IN+LeakyReLU → pool3 →
// [conv(k=5,p=2) → IN+LeakyReLU → pool3] ×2. Returns row-major (T_out, 60).
static bool pyannote_sincnet_ggml(pyannote_seg_context* ctx, ggml_backend_t be, const float* samples, int n_samples,
                                  std::vector<float>& out, int& T_out) {
    pyannote_seg_bench_stage _bs("sincnet_ggml");
    const auto& m = ctx->model;

    pyannote_graph_run run(96);
    if (!run.ctx0)
        return false;
    ggml_context* c = run.ctx0;

    ggml_tensor* audio = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_samples, 1);
    ggml_set_input(audio);

    auto inorm_lrelu = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        // x is (T, C): ggml_norm normalizes over ne[0]=T → InstanceNorm.
        if (w && b) {
            const int64_t C = x->ne[1];
            x = ggml_norm(c, x, 1e-5f);
            x = ggml_add(c, ggml_mul(c, x, ggml_reshape_2d(c, w, 1, C)), ggml_reshape_2d(c, b, 1, C));
        }
        return ggml_leaky_relu(c, x, 0.01f, false);
    };

    ggml_tensor* x = ggml_conv_1d(c, m.sinc_filters, audio, 10, 0, 1); // (T, 80)
    x = ggml_abs(c, x);                                                // SincNet uses |conv|
    x = inorm_lrelu(x, m.norm0_w, m.norm0_b);
    x = ggml_pool_1d(c, x, GGML_OP_POOL_MAX, 3, 3, 0); // (T/3, 80)

    x = ggml_conv_1d(c, m.conv1_w, x, 1, 2, 1); // (T1, 60)
    x = ggml_add(c, x, ggml_reshape_2d(c, m.conv1_b, 1, 60));
    x = inorm_lrelu(x, m.norm1_w, m.norm1_b);
    x = ggml_pool_1d(c, x, GGML_OP_POOL_MAX, 3, 3, 0);

    x = ggml_conv_1d(c, m.conv2_w, x, 1, 2, 1); // (T2, 60)
    x = ggml_add(c, x, ggml_reshape_2d(c, m.conv2_b, 1, 60));
    x = inorm_lrelu(x, m.norm2_w, m.norm2_b);
    x = ggml_pool_1d(c, x, GGML_OP_POOL_MAX, 3, 3, 0); // (T3, 60)

    ggml_tensor* res = ggml_cont(c, ggml_transpose(c, x)); // (60, T3) → row-major (T3, 60)
    if (!run.compute(be, res))
        return false;
    ggml_backend_tensor_set(audio, samples, 0, (size_t)n_samples * sizeof(float));
    if (ggml_backend_graph_compute(be, run.gf) != GGML_STATUS_SUCCESS)
        return false;

    T_out = (int)res->ne[1];
    out.resize((size_t)T_out * 60);
    ggml_backend_tensor_get(res, out.data(), 0, out.size() * sizeof(float));
    return true;
}

// One bidirectional LSTM layer. Input/output row-major (T, C_in) / (T, 2H).
// Input projections batched as one mul_mat per direction; recurrence in
// plain contiguous loops, one thread per direction.
static bool bilstm_forward_ggml(pyannote_seg_context* ctx, ggml_backend_t be, const float* input, int T, int C_in,
                                const pyannote_lstm& lstm, float* output) {
    const int H = lstm.hidden_size;

    // XW[dir] = W_dir @ x_t + Bi + Bh for all t: (4H, T) row-major (T, 4H).
    std::vector<float> xw[2];
    {
        pyannote_graph_run run(24);
        if (!run.ctx0)
            return false;
        ggml_context* c = run.ctx0;
        ggml_tensor* X = ggml_new_tensor_2d(c, GGML_TYPE_F32, C_in, T); // row-major (T, C_in)
        ggml_set_input(X);
        ggml_tensor* outs[2];
        for (int dir = 0; dir < 2; dir++) {
            // W ne=(C_in, 4H, 2) → dir slice (C_in, 4H); B ne=(8H, 2).
            ggml_tensor* Wd = ggml_view_2d(c, lstm.W, C_in, 4 * H, lstm.W->nb[1], (size_t)dir * lstm.W->nb[2]);
            ggml_tensor* Bi =
                ggml_cont(c, ggml_view_2d(c, lstm.B, 4 * H, 1, lstm.B->nb[1], (size_t)dir * lstm.B->nb[1]));
            ggml_tensor* Bh = ggml_cont(c, ggml_view_2d(c, lstm.B, 4 * H, 1, lstm.B->nb[1],
                                                        (size_t)dir * lstm.B->nb[1] + 4 * H * lstm.B->nb[0]));
            ggml_tensor* mm = ggml_mul_mat(c, ggml_cont(c, Wd), X); // (4H, T)
            outs[dir] = ggml_add(c, ggml_add(c, mm, Bi), Bh);
            ggml_set_output(outs[dir]);
            ggml_build_forward_expand(run.gf, outs[dir]);
        }
        run.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(run.ga, run.gf))
            return false;
        ggml_backend_tensor_set(X, input, 0, (size_t)T * C_in * sizeof(float));
        if (ggml_backend_graph_compute(be, run.gf) != GGML_STATUS_SUCCESS)
            return false;
        for (int dir = 0; dir < 2; dir++) {
            xw[dir].resize((size_t)T * 4 * H);
            ggml_backend_tensor_get(outs[dir], xw[dir].data(), 0, xw[dir].size() * sizeof(float));
        }
    }

    // Sequential recurrence — contiguous dots over H (autovectorized).
    //
    // This is the hot loop of the whole model: a 4H x H matvec per timestep,
    // per direction, per layer.
    //
    // ⚠ An F16 copy of R was tried here and LOST: 173.6 s of LSTM CPU time
    // against 146-162 s for F32, interleaved on a 2888 s file. The reasoning
    // that motivated it — "512x128 F32 restreamed 171k x 4 layers x 2
    // directions is ~350 GB, so halve it" — ignores that all of R is only 2 MB
    // and therefore L2-resident, so there was no DRAM bandwidth to win back,
    // while the per-element widening costs real cycles in the inner loop.
    // Don't re-try without a new argument.
    auto run_dir = [&](int dir) {
        const float* R_dir = (const float*)lstm.R->data + (size_t)dir * 4 * H * H;
        std::vector<float> h(H, 0.f), cc(H, 0.f), gates(4 * H);
        auto sigmoid = [](float x) { return 1.f / (1.f + expf(-x)); };
        for (int step = 0; step < T; step++) {
            int t = (dir == 0) ? step : (T - 1 - step);
            const float* xwt = xw[dir].data() + (size_t)t * 4 * H;
            for (int g = 0; g < 4 * H; g++) {
                const float* r = R_dir + (size_t)g * H;
                float sum = xwt[g];
                for (int j = 0; j < H; j++)
                    sum += r[j] * h[j];
                gates[g] = sum;
            }
            // ONNX gate order: i, o, f, c
            for (int j = 0; j < H; j++) {
                float i_g = sigmoid(gates[0 * H + j]);
                float o_g = sigmoid(gates[1 * H + j]);
                float f_g = sigmoid(gates[2 * H + j]);
                float c_g = tanhf(gates[3 * H + j]);
                cc[j] = f_g * cc[j] + i_g * c_g;
                h[j] = o_g * tanhf(cc[j]);
            }
            float* out_t = output + (size_t)t * 2 * H;
            for (int j = 0; j < H; j++)
                out_t[dir * H + j] = h[j];
        }
    };
    std::thread bwd(run_dir, 1);
    run_dir(0);
    bwd.join();
    return true;
}

// 3× Linear (+LeakyReLU) + LogSoftmax over the 7 classes.
// Input row-major (T, 256); output row-major (T, 7).
static bool pyannote_classifier_ggml(pyannote_seg_context* ctx, ggml_backend_t be, const float* input, int T,
                                     std::vector<float>& out) {
    pyannote_seg_bench_stage _bs("classifier_ggml");
    const auto& m = ctx->model;

    pyannote_graph_run run(32);
    if (!run.ctx0)
        return false;
    ggml_context* c = run.ctx0;

    ggml_tensor* X = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, T);
    ggml_set_input(X);
    // matmul weights stored ne=(C_out, C_in) — transpose for mul_mat's (k, n).
    auto lin = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, bool lrelu) {
        ggml_tensor* wt = ggml_cont(c, ggml_transpose(c, w));
        x = ggml_mul_mat(c, wt, x); // (C_out, T)
        if (b)
            x = ggml_add(c, x, b);
        return lrelu ? ggml_leaky_relu(c, x, 0.01f, false) : x;
    };
    ggml_tensor* x = lin(X, m.matmul0_w, m.linear0_b, true);
    x = lin(x, m.matmul1_w, m.linear1_b, true);
    x = lin(x, m.matmul2_w, m.linear2_b, false); // (7, T)
    x = ggml_log(c, ggml_soft_max(c, x));        // LogSoftmax over ne[0]=7

    if (!run.compute(be, x))
        return false;
    ggml_backend_tensor_set(X, input, 0, (size_t)T * 256 * sizeof(float));
    if (ggml_backend_graph_compute(be, run.gf) != GGML_STATUS_SUCCESS)
        return false;

    out.resize((size_t)T * 7);
    ggml_backend_tensor_get(x, out.data(), 0, out.size() * sizeof(float)); // (7,T) flat = row-major (T,7)
    return true;
}

// SincNet's downsampling: conv(stride 10) then three max-pools of stride 3.
// Output frame f is a function of input samples [270f, 270f + 511) — the 251
// -sample sinc receptive field plus the span the pools reach across. Both
// constants matter for chunking: kFrameStride makes chunk-local frames line up
// with whole-file frames exactly, and kFrameSpan says how much audio past the
// last wanted frame a chunk still has to be handed.
constexpr int kFrameStride = 270;
constexpr int kFrameSpan = 511;

static int pyannote_frames_for_samples(int n_samples) {
    if (n_samples < kFrameSpan)
        return 0;
    int t = (n_samples - 251) / 10 + 1; // conv0, stride 10, no padding
    return t / 3 / 3 / 3;               // three max-pools, stride 3
}

// Run the full net over one contiguous span of audio. `be` is the backend this
// call may use — chunk workers each get their own so they can run at once.
static bool pyannote_forward_span(pyannote_seg_context* ctx, ggml_backend_t be, const float* samples, int n_samples,
                                  std::vector<float>& logp, int& T) {
    const auto& m = ctx->model;

    std::vector<float> feats;
    T = 0;
    if (!pyannote_sincnet_ggml(ctx, be, samples, n_samples, feats, T) || T <= 0)
        return false;

    std::vector<float> lstm_out((size_t)T * 256);
    {
        pyannote_seg_bench_stage _bs("lstm_ggml");
        std::vector<float> lstm_in = std::move(feats);
        for (int li = 0; li < 4; li++) {
            int C_in = (li == 0) ? 60 : 256;
            if (!bilstm_forward_ggml(ctx, be, lstm_in.data(), T, C_in, m.lstm[li], lstm_out.data()))
                return false;
            lstm_in = lstm_out;
        }
    }
    return pyannote_classifier_ggml(ctx, be, lstm_out.data(), T, logp);
}

// ── Chunked parallel inference (#326) ──────────────────────────────────────
//
// The whole-file path fed every sample through as ONE sequence. That made the
// 4 stacked BiLSTMs a single scan of n_samples/270 timesteps, and since the
// recurrence is inherently sequential it ran on exactly two threads (one per
// direction) no matter what -t said. On a 48-minute file that is 171k steps
// and ~70% of segmentation time, with every core past the second one idle.
//
// So cut the audio into fixed-length chunks and run whole chunks concurrently,
// each on its own backend. Two things make this safe rather than merely fast:
//
//  * Chunk length is FIXED, never derived from the thread count, so the output
//    does not change when -t does. Thread count picks how many chunks are in
//    flight, nothing else.
//  * Each chunk is computed with kContextSeconds of real audio spliced on
//    either side and then trimmed away. That context absorbs the two edge
//    effects — the k=5 convolutions' zero padding, and the LSTM starting from
//    a zero hidden state — so interior frames land where the whole-file scan
//    put them.
//
// It also moves us TOWARD pyannote's own design rather than away from it:
// segmentation-3.0's 7 classes are a powerset over the speakers active in a
// *local* window, and upstream infers on a sliding 10 s window. One continuous
// 48-minute scan was the outlier.
struct pyannote_chunking {
    int chunk_frames = 0;   // frames emitted per chunk
    int context_frames = 0; // frames of real audio added each side, then dropped
    int n_workers = 1;
};

static bool pyannote_chunking_for(const pyannote_seg_context* ctx, int total_frames, pyannote_chunking& out) {
    // CRISPASR_PYANNOTE_CHUNK_S=0 restores the old single-scan behaviour.
    //
    // 60 s / 5 s picked on the VoxConverse dev shard (8 files, DER vs human
    // labels, no embedder re-clustering so the absolute numbers are the weak
    // raw-pyannote path):
    //
    //   whole-file 33.37% | 60/ctx2 32.88% | 60/ctx5 30.67% | 60/ctx10 31.87%
    //                     | 120/ctx5 31.08% | 120/ctx10 29.88%
    //
    // Every chunked setting beats the single scan, so this is not a speed-for-
    // accuracy trade. The spread among chunked settings is smaller than the
    // file-to-file spread on 8 files, so 120/ctx10 is not meaningfully better
    // than 60/ctx5 — and 60 s chunks parallelise on short and medium files
    // where 120 s would yield two chunks and half the cores idle. Context
    // costs (60+2*5)/60 = 17% redundant compute either way.
    double chunk_s = 60.0, context_s = 5.0;
    if (const char* e = crispasr_env::get("CRISPASR_PYANNOTE_CHUNK_S"))
        chunk_s = atof(e);
    if (const char* e = crispasr_env::get("CRISPASR_PYANNOTE_CHUNK_CONTEXT_S"))
        context_s = atof(e);
    if (chunk_s <= 0.0)
        return false;

    const double frames_per_s = 16000.0 / kFrameStride;
    out.chunk_frames = (int)(chunk_s * frames_per_s);
    out.context_frames = (int)(context_s * frames_per_s);
    if (out.chunk_frames <= 0)
        return false;
    // One chunk plus its context is the whole file — nothing to win, and the
    // single-scan path is then bit-identical to what shipped before.
    if (total_frames <= out.chunk_frames)
        return false;

    // Whether to chunk depends ONLY on how long the audio is — never on the
    // thread count. Deciding it on worker count would mean `-t 1` silently
    // took the single-scan path and produced different posteriors from `-t 2`
    // on the same file. Thread count picks how many chunks are in flight, and
    // nothing else: -t 2/4/8 are byte-identical, verified on a 2888 s file.
    const int n_chunks = (total_frames + out.chunk_frames - 1) / out.chunk_frames;
    out.n_workers = std::max(1, std::min(ctx->n_threads, n_chunks));
    return true;
}

static float* pyannote_seg_run_ggml(pyannote_seg_context* ctx, const float* samples, int n_samples, int* out_T) {
    const int total_frames = pyannote_frames_for_samples(n_samples);
    if (total_frames <= 0)
        return nullptr;

    pyannote_chunking ch;
    if (!pyannote_chunking_for(ctx, total_frames, ch)) {
        // Single scan over everything — the pre-#326 path, unchanged.
        std::vector<float> logp;
        int T = 0;
        if (!pyannote_forward_span(ctx, ctx->model.backend, samples, n_samples, logp, T) || T <= 0)
            return nullptr;
        if (out_T)
            *out_T = T;
        float* result = (float*)malloc((size_t)T * 7 * sizeof(float));
        if (result)
            memcpy(result, logp.data(), (size_t)T * 7 * sizeof(float));
        pyannote_seg_dump(result, T);
        return result;
    }

    const int n_chunks = (total_frames + ch.chunk_frames - 1) / ch.chunk_frames;
    if (pyannote_seg_bench_enabled())
        fprintf(stderr, "  pyannote_seg_bench: chunked %d frames -> %d chunks x %d frames, %d workers\n", total_frames,
                n_chunks, ch.chunk_frames, ch.n_workers);

    std::vector<float> result_buf((size_t)total_frames * 7);
    struct chunk_result {
        std::vector<float> logp; // (T, 7), starting at global frame g0 - pad_lo
        int pad_lo = 0;
        int T = 0;
    };
    std::vector<chunk_result> chunks((size_t)n_chunks);
    std::atomic<int> next_chunk{0};
    std::atomic<bool> failed{false};

    auto worker = [&](ggml_backend_t be) {
        for (;;) {
            const int ci = next_chunk.fetch_add(1);
            if (ci >= n_chunks || failed.load())
                return;

            const int g0 = ci * ch.chunk_frames;
            const int g1 = std::min(total_frames, g0 + ch.chunk_frames);

            // Widen by context on both sides, clipped to the real audio, and
            // snapped to frame boundaries so chunk-local frame j is exactly
            // global frame (s_start / kFrameStride) + j.
            const int pad_lo = std::min(ch.context_frames, g0);
            const int s_start = (g0 - pad_lo) * kFrameStride;
            const int want_end = (g1 + ch.context_frames) * kFrameStride + kFrameSpan;
            const int s_end = std::min(n_samples, want_end);
            if (s_end - s_start < kFrameSpan) {
                failed.store(true);
                return;
            }

            chunk_result cr;
            cr.pad_lo = pad_lo;
            if (!pyannote_forward_span(ctx, be, samples + s_start, s_end - s_start, cr.logp, cr.T)) {
                failed.store(true);
                return;
            }
            chunks[(size_t)ci] = std::move(cr);
        }
    };

    {
        pyannote_seg_bench_stage _bs("chunked_total");
        std::vector<ggml_backend_t> backends;
        for (int i = 0; i < ch.n_workers; i++) {
            // Each worker needs its own backend: ggml_backend_graph_compute is
            // not re-entrant on one instance. The model weights are read-only
            // and shared, so only the scratch buffers are duplicated.
            ggml_backend_t be = core_cpu_backend::init();
            if (!be) {
                failed.store(true);
                break;
            }
            // Hand each worker the cores that inter-chunk parallelism cannot
            // use. With more chunks than threads this is 1 and everything is
            // busy anyway; with FEWER chunks than threads — any short file —
            // pinning workers to 1 thread would leave most of the machine idle
            // (a 2-chunk file on 8 cores used 2). ggml partitions matmul and
            // conv by output row, so no output element is split across
            // threads and the result does not depend on this number; the
            // byte-identity of the posteriors across -t is asserted below by
            // measurement, not assumed.
            core_cpu_backend::set_n_threads(be, std::max(1, ctx->n_threads / ch.n_workers));
            backends.push_back(be);
        }
        if (!failed.load()) {
            std::vector<std::thread> pool;
            for (size_t i = 1; i < backends.size(); i++)
                pool.emplace_back(worker, backends[i]);
            if (!backends.empty())
                worker(backends[0]);
            for (auto& t : pool)
                t.join();
        }
        for (ggml_backend_t be : backends)
            ggml_backend_free(be);
    }

    if (failed.load())
        return nullptr;

    // ── Stitch, aligning each chunk's local speaker numbering to the last ──
    //
    // The 7 classes are a powerset over *local* speakers, and which physical
    // voice ends up as local spk0 vs spk1 is arbitrary per forward pass. Left
    // alone, every seam permutes the labels and the output is unusable
    // downstream. Each chunk was computed with `pad_lo` frames of context that
    // the PREVIOUS chunk already owns, so those frames are the same audio
    // scored twice — pick the relabelling of {spk0, spk1, spk2} that makes the
    // two agree best on them, then apply it to the frames this chunk owns.
    //
    // Alignment is sequential (chunk i aligns to the already-aligned i-1) but
    // it is pure arithmetic over a couple of seconds of frames, so it costs
    // nothing next to the forward passes, which all ran in parallel above.
    // class_map[p][c] = class c relabelled by permutation p (core/powerset.h).
    int class_map[6][core_powerset::kClasses];
    for (int pi = 0; pi < 6; pi++)
        for (int c = 0; c < core_powerset::kClasses; c++)
            class_map[pi][c] = core_powerset::permute_class(c, core_powerset::kPerms[pi]);

    for (int ci = 0; ci < n_chunks; ci++) {
        const chunk_result& cr = chunks[(size_t)ci];
        const int g0 = ci * ch.chunk_frames;
        const int g1 = std::min(total_frames, g0 + ch.chunk_frames);

        int best = 0; // identity
        if (ci > 0 && cr.pad_lo > 0) {
            double best_score = -1.0;
            for (int pi = 0; pi < 6; pi++) {
                double score = 0.0;
                for (int f = 0; f < cr.pad_lo; f++) {
                    const float* cur = &cr.logp[(size_t)f * 7];
                    const float* prev = &result_buf[(size_t)(g0 - cr.pad_lo + f) * 7];
                    for (int c = 0; c < 7; c++)
                        score += (double)expf(prev[class_map[pi][c]]) * expf(cur[c]);
                }
                if (score > best_score) {
                    best_score = score;
                    best = pi;
                }
            }
        }

        const int avail = cr.T - cr.pad_lo;
        const int keep = std::min(g1 - g0, std::max(0, avail));
        for (int f = 0; f < keep; f++) {
            const float* src = &cr.logp[(size_t)(cr.pad_lo + f) * 7];
            float* dst = &result_buf[(size_t)(g0 + f) * 7];
            for (int c = 0; c < 7; c++)
                dst[class_map[best][c]] = src[c];
        }
        // A short tail chunk can come up a frame or two shy once the trailing
        // context is clipped; repeat the last good frame rather than leaving
        // zeros, which would read as a confident silence.
        for (int f = keep; f < g1 - g0; f++)
            memcpy(&result_buf[(size_t)(g0 + f) * 7], keep > 0 ? &result_buf[(size_t)(g0 + keep - 1) * 7] : &cr.logp[0],
                   7 * sizeof(float));
    }

    if (out_T)
        *out_T = total_frames;
    float* result = (float*)malloc(result_buf.size() * sizeof(float));
    if (result)
        memcpy(result, result_buf.data(), result_buf.size() * sizeof(float));
    pyannote_seg_dump(result, total_frames);
    return result;
}

// ===========================================================================
// LSTM forward (bidirectional)
// ===========================================================================

// Run one bidirectional LSTM layer. Input: (T, C_in) row-major.
// Output: (T, 2*hidden) row-major (forward + backward concatenated).
static void bilstm_forward(const float* input, int T, int C_in, const pyannote_lstm& lstm, float* output) {
    const int H = lstm.hidden_size;
    const int dirs = 2;

    for (int dir = 0; dir < dirs; dir++) {
        // Extract per-direction weights
        // W: (2, 4H, C_in) → dir-slice: W[dir] = (4H, C_in)
        const float* W_dir = (const float*)lstm.W->data + (size_t)dir * 4 * H * C_in;
        const float* R_dir = (const float*)lstm.R->data + (size_t)dir * 4 * H * H;
        // B: (2, 8H) = (2, 4H input_bias + 4H hidden_bias)
        const float* Bi_dir = (const float*)lstm.B->data + (size_t)dir * 8 * H;
        const float* Bh_dir = Bi_dir + 4 * H;

        std::vector<float> h(H, 0.f), c(H, 0.f);
        std::vector<float> gates(4 * H);

        auto sigmoid = [](float x) { return 1.f / (1.f + expf(-x)); };

        for (int step = 0; step < T; step++) {
            int t = (dir == 0) ? step : (T - 1 - step);
            const float* x = input + (size_t)t * C_in;

            // gates = W @ x + R @ h + bias
            for (int g = 0; g < 4 * H; g++) {
                float sum = Bi_dir[g] + Bh_dir[g];
                for (int j = 0; j < C_in; j++)
                    sum += W_dir[g * C_in + j] * x[j];
                for (int j = 0; j < H; j++)
                    sum += R_dir[g * H + j] * h[j];
                gates[g] = sum;
            }

            // ONNX LSTM gate order is i, o, f, c (cell candidate), not
            // PyTorch's common i, f, g, o ordering.  See ONNX spec §LSTM.
            for (int j = 0; j < H; j++) {
                float i_g = sigmoid(gates[0 * H + j]);
                float o_g = sigmoid(gates[1 * H + j]);
                float f_g = sigmoid(gates[2 * H + j]);
                float c_g = tanhf(gates[3 * H + j]);
                c[j] = f_g * c[j] + i_g * c_g;
                h[j] = o_g * tanhf(c[j]);
            }

            // Write output: forward → first H, backward → second H
            float* out_t = output + (size_t)t * 2 * H;
            for (int j = 0; j < H; j++)
                out_t[dir * H + j] = h[j];
        }
    }
}

// ===========================================================================
// Public API
// ===========================================================================

// ===========================================================================
// onnxruntime path (CRISPASR_USE_ONNXRUNTIME) — model path ending in ".onnx".
// The sherpa-onnx pyannote-segmentation-3-0 export takes raw 16 kHz mono PCM
// as input ("x", [N, 1, T]) and returns per-frame log-softmax posteriors
// ("y", [N, T_out, 7]) — the same contract as pyannote_seg_run().
// ===========================================================================

#if defined(CRISPASR_USE_ONNXRUNTIME)

static struct pyannote_seg_context* pyannote_seg_init_ort(struct pyannote_seg_context* ctx, const char* model_path,
                                                          int n_threads) {
    const bool force_cpu = crispasr_env::get("CRISPASR_ORT_FORCE_CPU") != nullptr;
    std::string provider;
    try {
        ctx->ort_session = new Ort::Session(
            crispasr_ort::create_session(model_path, n_threads > 0 ? n_threads : 4, force_cpu, provider));
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "pyannote_seg[ort]: failed to create session: %s\n", e.what());
        delete ctx;
        return nullptr;
    }
    fprintf(stderr, "pyannote_seg[ort]: %s (%s EP)\n", model_path, provider.c_str());
    return ctx;
}

static float* pyannote_seg_run_ort(pyannote_seg_context* ctx, const float* samples, int n_samples, int* out_T) {
    std::vector<int64_t> shape = {1, 1, n_samples};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(samples), (size_t)n_samples, shape.data(), 3);
    const char* in_names[] = {"x"};
    const char* out_names[] = {"y"};
    Ort::RunOptions ro;
    auto outs = ctx->ort_session->Run(ro, in_names, &in, 1, out_names, 1);
    if (outs.size() != 1)
        return nullptr;
    auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (oshape.size() != 3)
        return nullptr;
    const int T_out = (int)oshape[1];
    if (T_out <= 0)
        return nullptr;
    float* result = (float*)malloc((size_t)T_out * 7 * sizeof(float));
    if (!result)
        return nullptr;
    memcpy(result, outs[0].GetTensorData<float>(), (size_t)T_out * 7 * sizeof(float));
    *out_T = T_out;
    return result;
}

#endif // CRISPASR_USE_ONNXRUNTIME

extern "C" struct pyannote_seg_context* pyannote_seg_init(const char* gguf_path, int n_threads) {
    auto* ctx = new pyannote_seg_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

#if defined(CRISPASR_USE_ONNXRUNTIME)
    if (crispasr_ort::is_onnx_path(gguf_path))
        return pyannote_seg_init_ort(ctx, gguf_path, n_threads);
#endif

    if (!pyannote_load(ctx->model, gguf_path)) {
        delete ctx;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(ctx->model.backend, ctx->n_threads);
    fprintf(stderr, "pyannote_seg: loaded (%d LSTM layers, hidden=%d)\n", 4, ctx->model.lstm[0].hidden_size);
    return ctx;
}

extern "C" void pyannote_seg_free(struct pyannote_seg_context* ctx) {
    if (!ctx)
        return;
#if defined(CRISPASR_USE_ONNXRUNTIME)
    delete ctx->ort_session;
    ctx->ort_session = nullptr;
#endif
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->model.backend)
        ggml_backend_free(ctx->model.backend);
    delete ctx;
}

extern "C" float* pyannote_seg_run(struct pyannote_seg_context* ctx, const float* samples, int n_samples, int* out_T) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    pyannote_seg_bench_stage _bs_total("run_total");
#if defined(CRISPASR_USE_ONNXRUNTIME)
    if (ctx->ort_session)
        return pyannote_seg_run_ort(ctx, samples, n_samples, out_T);
#endif
    if (!pyannote_use_legacy())
        return pyannote_seg_run_ggml(ctx, samples, n_samples, out_T);
    const auto& m = ctx->model;

    // ---- SincNet front-end ----
    // Conv1d(1→80, kernel=251, stride=10) using the learned sinc filters
    const int K0 = 251, S0 = 10;
    const int C0 = 80;
    int T = (n_samples - K0) / S0 + 1;
    const float* sinc = (const float*)m.sinc_filters->data;

    std::vector<float> feat(C0 * T);
    for (int co = 0; co < C0; co++) {
        for (int t = 0; t < T; t++) {
            float sum = 0;
            for (int k = 0; k < K0; k++)
                sum += sinc[co * K0 + k] * samples[t * S0 + k];
            feat[co * T + t] = std::abs(sum); // SincNet uses |conv|
        }
    }

    // InstanceNorm + LeakyReLU after sinc
    // TODO: proper InstanceNorm; for now just normalize per-channel
    if (m.norm0_w) {
        const float* nw = (const float*)m.norm0_w->data;
        const float* nb = (const float*)m.norm0_b->data;
        for (int c = 0; c < C0; c++) {
            float mean = 0, var = 0;
            for (int t = 0; t < T; t++)
                mean += feat[c * T + t];
            mean /= T;
            for (int t = 0; t < T; t++) {
                float d = feat[c * T + t] - mean;
                var += d * d;
            }
            var /= T;
            float inv = 1.f / sqrtf(var + 1e-5f);
            for (int t = 0; t < T; t++)
                feat[c * T + t] = nw[c] * (feat[c * T + t] - mean) * inv + nb[c];
        }
    }
    // LeakyReLU
    for (float& v : feat)
        v = v >= 0 ? v : 0.01f * v;

    // MaxPool1d(kernel=3, stride=3)
    int T1 = T / 3;
    std::vector<float> pool1(C0 * T1);
    for (int c = 0; c < C0; c++)
        for (int t = 0; t < T1; t++) {
            float mx = feat[c * T + t * 3];
            for (int k = 1; k < 3; k++)
                mx = std::max(mx, feat[c * T + t * 3 + k]);
            pool1[c * T1 + t] = mx;
        }

    // Conv1d(80→60, k=5, pad=2) + InstanceNorm + LeakyReLU + MaxPool(3)
    auto do_conv_norm_pool = [&](const std::vector<float>& in, int C_in, int T_in, ggml_tensor* cw, ggml_tensor* cb,
                                 ggml_tensor* nw, ggml_tensor* nb, int C_out, int K, int pool_k,
                                 std::vector<float>& out, int& T_out) {
        int pad = K / 2;
        std::vector<float> conv_out(C_out * T_in);
        const float* w = (const float*)cw->data;
        const float* b = (const float*)cb->data;
        for (int co = 0; co < C_out; co++) {
            for (int t = 0; t < T_in; t++) {
                float sum = b[co];
                for (int ci = 0; ci < C_in; ci++)
                    for (int k = 0; k < K; k++) {
                        int ti = t + k - pad;
                        if (ti >= 0 && ti < T_in)
                            sum += w[(co * C_in + ci) * K + k] * in[ci * T_in + ti];
                    }
                conv_out[co * T_in + t] = sum;
            }
        }
        // InstanceNorm
        if (nw && nb) {
            const float* nwp = (const float*)nw->data;
            const float* nbp = (const float*)nb->data;
            for (int c = 0; c < C_out; c++) {
                float mean = 0, var = 0;
                for (int t = 0; t < T_in; t++)
                    mean += conv_out[c * T_in + t];
                mean /= T_in;
                for (int t = 0; t < T_in; t++) {
                    float d = conv_out[c * T_in + t] - mean;
                    var += d * d;
                }
                var /= T_in;
                float inv = 1.f / sqrtf(var + 1e-5f);
                for (int t = 0; t < T_in; t++)
                    conv_out[c * T_in + t] = nwp[c] * (conv_out[c * T_in + t] - mean) * inv + nbp[c];
            }
        }
        // LeakyReLU
        for (float& v : conv_out)
            v = v >= 0 ? v : 0.01f * v;
        // MaxPool
        T_out = T_in / pool_k;
        out.resize(C_out * T_out);
        for (int c = 0; c < C_out; c++)
            for (int t = 0; t < T_out; t++) {
                float mx = conv_out[c * T_in + t * pool_k];
                for (int k = 1; k < pool_k; k++)
                    mx = std::max(mx, conv_out[c * T_in + t * pool_k + k]);
                out[c * T_out + t] = mx;
            }
    };

    std::vector<float> stage1_out;
    int T2;
    do_conv_norm_pool(pool1, C0, T1, m.conv1_w, m.conv1_b, m.norm1_w, m.norm1_b, 60, 5, 3, stage1_out, T2);

    std::vector<float> stage2_out;
    int T3;
    do_conv_norm_pool(stage1_out, 60, T2, m.conv2_w, m.conv2_b, m.norm2_w, m.norm2_b, 60, 5, 3, stage2_out, T3);

    // Transpose to (T, C=60) for LSTM input
    std::vector<float> lstm_in(T3 * 60);
    for (int t = 0; t < T3; t++)
        for (int c = 0; c < 60; c++)
            lstm_in[t * 60 + c] = stage2_out[c * T3 + t];

    // ---- 4× bidirectional LSTM ----
    int T_lstm = T3;
    std::vector<float> lstm_out;
    for (int li = 0; li < 4; li++) {
        int C_in = (li == 0) ? 60 : 256;
        lstm_out.resize(T_lstm * 256);
        bilstm_forward(lstm_in.data(), T_lstm, C_in, m.lstm[li], lstm_out.data());
        lstm_in = lstm_out;
    }

    // ---- 3× Linear + LeakyReLU + LogSoftmax ----
    // Linear 0: (256→128) + LeakyReLU
    auto linear = [](const float* in, int T, int C_in, const float* w, const float* b, int C_out, float* out,
                     bool leaky_relu = true) {
        for (int t = 0; t < T; t++) {
            for (int co = 0; co < C_out; co++) {
                float sum = b ? b[co] : 0.f;
                for (int ci = 0; ci < C_in; ci++)
                    sum += w[ci * C_out + co] * in[t * C_in + ci];
                out[t * C_out + co] = leaky_relu ? (sum >= 0 ? sum : 0.01f * sum) : sum;
            }
        }
    };

    std::vector<float> lin0(T_lstm * 128);
    linear(lstm_out.data(), T_lstm, 256, (const float*)m.matmul0_w->data,
           m.linear0_b ? (const float*)m.linear0_b->data : nullptr, 128, lin0.data());

    std::vector<float> lin1(T_lstm * 128);
    linear(lin0.data(), T_lstm, 128, (const float*)m.matmul1_w->data,
           m.linear1_b ? (const float*)m.linear1_b->data : nullptr, 128, lin1.data());

    std::vector<float> lin2(T_lstm * 7);
    linear(lin1.data(), T_lstm, 128, (const float*)m.matmul2_w->data,
           m.linear2_b ? (const float*)m.linear2_b->data : nullptr, 7, lin2.data(),
           false); // no LeakyReLU before LogSoftmax

    // LogSoftmax over the 7 classes per frame
    for (int t = 0; t < T_lstm; t++) {
        float* lv = lin2.data() + t * 7;
        float mx = *std::max_element(lv, lv + 7);
        float sum = 0;
        for (int i = 0; i < 7; i++) {
            lv[i] = expf(lv[i] - mx);
            sum += lv[i];
        }
        for (int i = 0; i < 7; i++)
            lv[i] = logf(lv[i] / sum);
    }

    if (out_T)
        *out_T = T_lstm;
    float* result = (float*)malloc(T_lstm * 7 * sizeof(float));
    if (result)
        memcpy(result, lin2.data(), T_lstm * 7 * sizeof(float));
    pyannote_seg_dump(result, T_lstm);
    return result;
}
