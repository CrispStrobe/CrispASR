// onsets_and_frames.cpp — Onsets & Frames piano transcription backend.
//
// See onsets_and_frames.h for the architecture and for why the ConvStack lives
// in a ggml graph while the BiLSTM does not. Conventions follow
// src/piano_transcription.cpp, which is Kong's CRNN in the same shape.

#include "onsets_and_frames.h"

#include "core/crispasr_env.h"
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "core/ggml_cpu_backend.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/mel.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

// ─── Hyperparameters ────────────────────────────────────────────────────────

struct oaf_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_fft = 2048;
    uint32_t hop_size = 512;
    uint32_t n_mels = 229;
    float fmin = 30.0f;
    float fmax = 8000.0f;
    uint32_t classes_num = 88;
    uint32_t begin_note = 21; // MIDI A0
    uint32_t lstm_hidden = 384;
    uint32_t fc_out = 768;
    uint32_t midfeat = 5472; // 96 channels × (229 // 4)
};

// log(clamp(mel, 1e-5)) — the reference's floor, not librosa's 1e-10.
static constexpr float OAF_MEL_FLOOR = 1e-5f;

// ONNX LSTM gate order is i, o, f, c — NOT PyTorch's i, f, g, o. The converter
// stores the weights in ONNX order and records the string in the GGUF; these
// offsets are the other half of that contract and are checked at load.
static constexpr int OAF_GATE_I = 0;
static constexpr int OAF_GATE_O = 1;
static constexpr int OAF_GATE_F = 2;
static constexpr int OAF_GATE_C = 3;

// Time chunk for the convolution stack. The three 3×3 convolutions reach one
// frame each, so a 3-frame halo makes a chunk bit-identical to the unchunked
// result everywhere but the true sequence edges, where the real zero padding
// applies and the halo is absent by construction.
static constexpr int OAF_CONV_HALO = 3;
static constexpr int OAF_CONV_CHUNK = 256;

// Timestep chunk for the graph BiLSTM. The recurrence is unrolled into the
// graph at ~16 nodes per step, so node count -- and with it the metadata arena,
// which is ggml_tensor_overhead() per node -- scales linearly in T. Chunking
// with a carried (h, c) bounds it at a fixed cost and is EXACT, because the
// recurrence is exact: chunk k starts from the state chunk k-1 ended with.
// 512 steps is ~8200 nodes, ~6 MB of metadata, on a box with 7 GB.
static constexpr int OAF_LSTM_GRAPH_CHUNK = 512;

// ─── Weights ────────────────────────────────────────────────────────────────

// One direction of an LSTM. W is consumed by ggml_mul_mat and therefore stays
// in whatever type the GGUF holds (q4_0 included); R is used by the per-step
// recurrence in plain C++ and is dequantised once at load.
struct oaf_lstm_dir {
    ggml_tensor* W = nullptr;   // [in, 4H] in ggml ne order
    ggml_tensor* R_t = nullptr; // the same recurrence matrix as a ggml tensor, [H, 4H] in ne order
    std::vector<float> R;       // [4H, H] row-major
    std::vector<float> b;       // [4H]
};

struct oaf_lstm {
    oaf_lstm_dir fwd;
    oaf_lstm_dir rev;
    int input_size = 0;
};

struct oaf_conv_stack {
    ggml_tensor* conv_w[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* conv_b[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* fc_w = nullptr;
    ggml_tensor* fc_b = nullptr;
};

struct oaf_head {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct oaf_weights {
    // Four ConvStacks, all reading the same mel.
    oaf_conv_stack onset_cs, offset_cs, activation_cs, velocity_cs;
    oaf_lstm onset_lstm, offset_lstm, frame_lstm;
    oaf_head onset_head, offset_head, activation_head, velocity_head, frame_head;
    ggml_tensor* mel_fb = nullptr; // [n_freqs, n_mels] in ggml ne order
    ggml_tensor* window = nullptr; // [n_fft]
};

struct onsets_and_frames_ctx {
    oaf_hparams hp;
    oaf_weights weights;
    onsets_and_frames_params params;

    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    ggml_backend_t backend = nullptr;

    std::vector<float> mel_fb; // [n_mels * n_freqs], MelsFreqs layout
    std::vector<float> hann;   // [n_fft]
    std::vector<uint8_t> graph_meta;

    // One allocator per graph shape, kept for the life of the context, on the
    // src/hft_transformer.cpp:108-115 pattern — whose own comment names THIS
    // file's fresh-allocator-per-chunk as the bug it exists to avoid.
    //
    // Before this, ggml_gallocr_new/_free ran eight-plus times per forward: once
    // per conv-stack chunk (inside the per-chunk loop), twice per BiLSTM
    // direction, and once per head. That is the #132 pattern documented at
    // src/crispasr.cpp:197-203 — per-call disposable allocators fragmenting
    // memory and costing 2-5x. A gallocr keeps its buffer when the next graph
    // fits, so reuse costs nothing and the arena grows monotonically.
    //
    // gallocr, NOT a scheduler, and the cgraph is rebuilt every call rather
    // than cached: caching a cgraph across calls on a shared sched leaves the
    // previous allocation's buffer/data on the cached tensors and the second
    // call onward reads stale memory (core/step_graph_cache.h:66-70, #208).
    ggml_gallocr_t alloc_conv = nullptr; // conv-stack chunk
    ggml_gallocr_t alloc_proj = nullptr; // LSTM input projection
    ggml_gallocr_t alloc_head = nullptr; // output head
    ggml_gallocr_t alloc_lstm = nullptr; // unrolled BiLSTM recurrence (CRISPASR_OAF_GRAPH_LSTM)

    // Separate from graph_meta: the unrolled recurrence needs megabytes of
    // tensor overhead where the other three shapes need kilobytes, and sharing
    // one arena would hold the big allocation for the life of the context.
    std::vector<uint8_t> lstm_meta;
};

// ─── Bench instrumentation (docs/contributing.md §1) ────────────────────────

static bool oaf_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("ONSETS_AND_FRAMES_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct oaf_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit oaf_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~oaf_bench_stage() {
        if (!oaf_bench_enabled())
            return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  oaf_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ─── Small helpers ──────────────────────────────────────────────────────────

static void oaf_fft_r2c(const float* in, int N, float* out) {
    std::vector<float> re(N), im(N, 0.0f);
    std::memcpy(re.data(), in, N * sizeof(float));
    core_fft::fft_radix2_inplace(re.data(), im.data(), N);
    for (int i = 0; i < N; i++) {
        out[2 * i + 0] = re[i];
        out[2 * i + 1] = im[i];
    }
}

// Dequantise any GGUF tensor to F32. Unlike piano_transcription's
// tensor_to_f32 this goes through ggml's type traits, so a q4_0 / q8_0 GGUF
// loads the same way an F32 one does — the quantised file is the point of the
// port and a loader that only understood F32/F16 would defeat it.
//
// `t->data` is only a dereferenceable host pointer when the tensor lives in a
// host buffer. That held while this file hard-coded the CPU backend; it does
// not once the weights can land in a Metal/CUDA/Vulkan buffer. Stage through
// ggml_backend_tensor_get(), which every backend implements. This matters here
// beyond the mel filterbank and window: the BiLSTM recurrence runs on the host
// (see the ConvStack/BiLSTM split in the header), so R and b are read back
// every load.
static std::vector<float> oaf_to_f32(const ggml_tensor* t) {
    const int64_t n = t ? ggml_nelements(t) : 0;
    std::vector<float> out((size_t)n);
    if (!t)
        return out;
    const bool host = !t->buffer || ggml_backend_buffer_is_host(t->buffer);
    std::vector<uint8_t> staged;
    const void* src = t->data;
    if (!host) {
        staged.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, staged.data(), 0, staged.size());
        src = staged.data();
    }
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), src, (size_t)n * sizeof(float));
        return out;
    }
    const ggml_type_traits* tr = ggml_get_type_traits(t->type);
    if (tr && tr->to_float) {
        tr->to_float(src, out.data(), n);
    } else {
        std::fprintf(stderr, "oaf: no dequantiser for tensor type %s\n", ggml_type_name(t->type));
    }
    return out;
}

static int oaf_nthreads(const onsets_and_frames_ctx* ctx) {
    if (ctx && ctx->params.n_threads > 0)
        return ctx->params.n_threads;
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : (int)std::min(hw, 8u);
}

// core_cpu_backend::set_n_threads() is ggml_backend_cpu_set_n_threads() in the
// ordinary (non-DL) build, and that one asserts ggml_backend_is_cpu(). Calling
// it on a Metal backend aborts the process, so every site is guarded now that
// ctx->backend may not be the CPU.
static inline void oaf_set_threads(const onsets_and_frames_ctx* ctx) {
    if (core_cpu_backend::is_cpu(ctx->backend))
        core_cpu_backend::set_n_threads(ctx->backend, oaf_nthreads(ctx));
}

static inline float oaf_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ─── Weight loading ─────────────────────────────────────────────────────────

static bool load_conv_stack(core_gguf::tensor_map& tm, const std::string& p, oaf_conv_stack& cs) {
    for (int i = 0; i < 3; i++) {
        cs.conv_w[i] = core_gguf::require(tm, (p + ".conv" + std::to_string(i) + ".weight").c_str(), "oaf");
        cs.conv_b[i] = core_gguf::require(tm, (p + ".conv" + std::to_string(i) + ".bias").c_str(), "oaf");
        if (!cs.conv_w[i] || !cs.conv_b[i])
            return false;
    }
    cs.fc_w = core_gguf::require(tm, (p + ".fc.weight").c_str(), "oaf");
    cs.fc_b = core_gguf::require(tm, (p + ".fc.bias").c_str(), "oaf");
    return cs.fc_w && cs.fc_b;
}

static bool load_lstm_dir(core_gguf::tensor_map& tm, const std::string& p, oaf_lstm_dir& d) {
    d.W = core_gguf::require(tm, (p + ".W").c_str(), "oaf");
    ggml_tensor* R = core_gguf::require(tm, (p + ".R").c_str(), "oaf");
    ggml_tensor* b = core_gguf::require(tm, (p + ".b").c_str(), "oaf");
    if (!d.W || !R || !b)
        return false;
    d.R_t = R; // lives in ctx->w_ctx; the graph recurrence consumes it directly
    d.R = oaf_to_f32(R);
    d.b = oaf_to_f32(b);
    return true;
}

static bool load_lstm(core_gguf::tensor_map& tm, const std::string& p, oaf_lstm& l) {
    if (!load_lstm_dir(tm, p + ".lstm.fwd", l.fwd) || !load_lstm_dir(tm, p + ".lstm.rev", l.rev))
        return false;
    l.input_size = (int)l.fwd.W->ne[0];
    return true;
}

static bool load_head(core_gguf::tensor_map& tm, const std::string& p, oaf_head& h) {
    h.w = core_gguf::require(tm, (p + ".head.weight").c_str(), "oaf");
    h.b = core_gguf::require(tm, (p + ".head.bias").c_str(), "oaf");
    return h.w && h.b;
}

struct onsets_and_frames_params onsets_and_frames_default_params(void) {
    return {
        /* n_threads       */ 4,
        /* verbosity       */ 1,
        /* use_gpu         */ false,
        /* onset_threshold */ 0.5f,
        /* frame_threshold */ 0.5f,
        /* segment_seconds */ 0.0f,
    };
}

struct onsets_and_frames_ctx* onsets_and_frames_init_from_file(const char* path,
                                                               struct onsets_and_frames_params params) {
    auto* ctx = new onsets_and_frames_ctx();
    ctx->params = params;

    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete ctx;
        return nullptr;
    }
    auto& hp = ctx->hp;
    hp.sample_rate = core_gguf::kv_u32(meta, "oaf.sample_rate", hp.sample_rate);
    hp.n_fft = core_gguf::kv_u32(meta, "oaf.n_fft", hp.n_fft);
    hp.hop_size = core_gguf::kv_u32(meta, "oaf.hop_size", hp.hop_size);
    hp.n_mels = core_gguf::kv_u32(meta, "oaf.n_mels", hp.n_mels);
    hp.fmin = core_gguf::kv_f32(meta, "oaf.fmin", hp.fmin);
    hp.fmax = core_gguf::kv_f32(meta, "oaf.fmax", hp.fmax);
    hp.classes_num = core_gguf::kv_u32(meta, "oaf.classes_num", hp.classes_num);
    hp.begin_note = core_gguf::kv_u32(meta, "oaf.begin_note", hp.begin_note);
    hp.lstm_hidden = core_gguf::kv_u32(meta, "oaf.lstm_hidden", hp.lstm_hidden);
    hp.fc_out = core_gguf::kv_u32(meta, "oaf.fc_out", hp.fc_out);
    hp.midfeat = core_gguf::kv_u32(meta, "oaf.midfeat", hp.midfeat);

    // The converter and this file must agree about the LSTM gate order; a
    // disagreement produces a model that runs and emits plausible garbage.
    const std::string gate_order = core_gguf::kv_str(meta, "oaf.lstm_gate_order", "iofc");

    // The dtypes actually present in this file, collected while the metadata
    // context is still open, so the GPU-capability probe below can ask about
    // the kernels this model will really need rather than a guess.
    std::vector<ggml_type> gguf_types;
    for (int64_t i = 0, nt = gguf_get_n_tensors(meta); i < nt; i++) {
        const ggml_type t = gguf_get_tensor_type(meta, i);
        if (std::find(gguf_types.begin(), gguf_types.end(), t) == gguf_types.end())
            gguf_types.push_back(t);
    }
    core_gguf::free_metadata(meta);
    if (gate_order != "iofc") {
        std::fprintf(stderr, "oaf: GGUF declares LSTM gate order '%s'; this runtime implements 'iofc'\n",
                     gate_order.c_str());
        delete ctx;
        return nullptr;
    }

    // CUDA > Metal > Vulkan > CPU, the src/crepe.cpp:346 pattern (issue #214).
    //
    // Unlike hFT this one is NOT obviously a GPU win and the wiring does not
    // assume it is. O&F is ~46% convolution and ~29% LSTM by FLOP, and the
    // LSTM here is a host-side sequential recurrence (see the ConvStack/BiLSTM
    // split in the header) that a GPU cannot help with — it only adds a
    // device↔host copy per chunk boundary. The ConvStack and the two GEMMs are
    // what can move. Whether that nets out positive is a measurement, not an
    // assumption; see docs/music-transcription/PIANO_METAL_AB.md.
    //
    // Two knobs, deliberately distinct:
    //   * params.use_gpu — the caller's intent. Already in the struct, never
    //     read until now; honoured rather than deleted.
    //   * CRISPASR_OAF_NO_GPU=1 — forces CPU whatever the caller asked, so one
    //     binary can run both A/B arms.
    const bool no_gpu = crispasr_env::get("CRISPASR_OAF_NO_GPU") != nullptr || !params.use_gpu;
    ctx->backend = no_gpu ? nullptr : crispasr_init_gpu_backend();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    if (!ctx->backend) {
        std::fprintf(stderr, "oaf: no ggml backend could be initialised\n");
        delete ctx;
        return nullptr;
    }
    // Ask the device rather than inferring from "the GPU call returned
    // non-null": crispasr_init_gpu_backend() ends in ggml_backend_init_best(),
    // which hands back the CPU backend on a CPU-only build.
    // A GPU that cannot MUL_MAT is worse than no GPU: this model drives a single
    // backend through ggml_gallocr, so an op the device declines aborts the
    // process rather than falling back. GitHub's hosted macos-14 runner is
    // exactly that case -- an "Apple Paravirtual device" with simdgroup matrix
    // multiply disabled, where the first encoder GEMM died with "unsupported op
    // 'MUL_MAT'". Ask first, and degrade to the CPU instead.
    if (!core_cpu_backend::is_cpu(ctx->backend) && !crispasr_backend_supports_mul_mat(ctx->backend, gguf_types)) {
        ggml_backend_free(ctx->backend);
        ctx->backend = core_cpu_backend::init();
        if (!ctx->backend) {
            delete ctx;
            return nullptr;
        }
    }
    if (params.verbosity >= 1)
        std::fprintf(stderr, "oaf: backend = %s (%s)\n", ggml_backend_name(ctx->backend),
                     core_cpu_backend::is_cpu(ctx->backend) ? "CPU" : "GPU");

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "oaf", wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;

    auto& w = ctx->weights;
    auto& tm = wl.tensors;
    bool ok = true;
    ok = ok && load_conv_stack(tm, "oaf.onset", w.onset_cs);
    ok = ok && load_conv_stack(tm, "oaf.offset", w.offset_cs);
    ok = ok && load_conv_stack(tm, "oaf.activation", w.activation_cs);
    ok = ok && load_conv_stack(tm, "oaf.velocity", w.velocity_cs);
    ok = ok && load_lstm(tm, "oaf.onset", w.onset_lstm);
    ok = ok && load_lstm(tm, "oaf.offset", w.offset_lstm);
    ok = ok && load_lstm(tm, "oaf.frame", w.frame_lstm);
    ok = ok && load_head(tm, "oaf.onset", w.onset_head);
    ok = ok && load_head(tm, "oaf.offset", w.offset_head);
    ok = ok && load_head(tm, "oaf.activation", w.activation_head);
    ok = ok && load_head(tm, "oaf.velocity", w.velocity_head);
    ok = ok && load_head(tm, "oaf.frame", w.frame_head);
    if (!ok) {
        onsets_and_frames_free(ctx);
        return nullptr;
    }

    w.mel_fb = core_gguf::try_get(tm, "oaf.mel_fb");
    w.window = core_gguf::try_get(tm, "oaf.window");

    const int n_freqs = (int)hp.n_fft / 2 + 1;
    if (w.mel_fb) {
        // Stored [n_mels, n_freqs] row-major, which is core_mel's MelsFreqs.
        ctx->mel_fb = oaf_to_f32(w.mel_fb);
    } else {
        // No fallback that is worth having: the checkpoint was trained on one
        // particular filterbank (HTK scale, slaney norm) and neither
        // build_htk_fb (no norm) nor build_slaney_fb (Slaney scale) is it.
        std::fprintf(stderr, "oaf: GGUF has no oaf.mel_fb — reconvert with "
                             "models/convert-onsets-and-frames-to-gguf.py\n");
        onsets_and_frames_free(ctx);
        return nullptr;
    }
    if ((int)ctx->mel_fb.size() != (int)hp.n_mels * n_freqs) {
        std::fprintf(stderr, "oaf: mel_fb is %zu floats, expected %d\n", ctx->mel_fb.size(), (int)hp.n_mels * n_freqs);
        onsets_and_frames_free(ctx);
        return nullptr;
    }
    if (w.window) {
        ctx->hann = oaf_to_f32(w.window);
    } else {
        ctx->hann.resize(hp.n_fft);
        for (uint32_t i = 0; i < hp.n_fft; i++)
            ctx->hann[i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)i / (float)hp.n_fft);
    }

    if (params.verbosity >= 1) {
        std::fprintf(stderr, "oaf: loaded model (%u mels, %u classes, hop %u, lstm %u)\n", hp.n_mels, hp.classes_num,
                     hp.hop_size, hp.lstm_hidden);
    }
    return ctx;
}

void onsets_and_frames_free(struct onsets_and_frames_ctx* ctx) {
    if (!ctx)
        return;
    if (ctx->alloc_conv)
        ggml_gallocr_free(ctx->alloc_conv);
    if (ctx->alloc_proj)
        ggml_gallocr_free(ctx->alloc_proj);
    if (ctx->alloc_head)
        ggml_gallocr_free(ctx->alloc_head);
    if (ctx->alloc_lstm)
        ggml_gallocr_free(ctx->alloc_lstm);
    if (ctx->w_buf)
        core_gguf::release_weight_buffer(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

uint32_t onsets_and_frames_sample_rate(const struct onsets_and_frames_ctx* ctx) {
    return ctx ? ctx->hp.sample_rate : 16000;
}

// ─── Front end ──────────────────────────────────────────────────────────────

// Returns [T, n_mels] row-major, log(clamp(mel, 1e-5)).
static std::vector<float> oaf_log_mel(onsets_and_frames_ctx* ctx, const float* pcm, int n_samples, int& T_out) {
    oaf_bench_stage _b("mel");
    auto& hp = ctx->hp;
    const int n_freqs = (int)hp.n_fft / 2 + 1;

    // The reference forward does `x[:, :-1]` before the STFT; that single
    // dropped sample is what makes the frame count come out at
    // (len - 1) // hop + 1 rather than len // hop + 1, and a one-frame shift
    // in the middle of a piece is a 32 ms timing error on every note.
    const int n_used = std::max(0, n_samples - 1);

    core_mel::Params p;
    p.n_fft = (int)hp.n_fft;
    p.hop_length = (int)hp.hop_size;
    p.win_length = (int)hp.n_fft;
    p.n_mels = (int)hp.n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = OAF_MEL_FLOOR;
    p.spec_kind = core_mel::SpecKind::Magnitude; // power 1.0 — NOT |X|²
    p.norm = core_mel::Normalization::None;
    p.layout = core_mel::Layout::TimeMels;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Float;
    p.center_pad = true;
    p.center_pad_reflect = true;

    return core_mel::compute(pcm, n_used, ctx->hann.data(), (int)hp.n_fft, ctx->mel_fb.data(), n_freqs, oaf_fft_r2c, p,
                             T_out);
}

float* onsets_and_frames_mel(struct onsets_and_frames_ctx* ctx, const float* pcm, int n_samples, int* out_frames) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;
    int T = 0;
    auto mel = oaf_log_mel(ctx, pcm, n_samples, T);
    if (out_frames)
        *out_frames = T;
    if (mel.empty())
        return nullptr;
    float* buf = (float*)std::malloc(mel.size() * sizeof(float));
    if (!buf)
        return nullptr;
    std::memcpy(buf, mel.data(), mel.size() * sizeof(float));
    return buf;
}

// ─── Per-layer stage capture (the diff harness) ─────────────────────────────
//
// O&F was the one model of the six with no per-layer parity path at all — what
// it had (tests/oaf_parity_dump.cpp + tools/oaf_parity.py) compares a mel and
// five heads, so a regression inside a ConvStack or a BiLSTM surfaces as "the
// onset head moved" rather than as a layer. These taps are what localise it.
//
// Every captured tensor is marked with BOTH ggml_set_name and ggml_set_output:
// without the latter gallocr reuses the memory and a later layer overwrites the
// value (playbook §5.0/§6.6). Capture is opt-in — a null sink costs one branch
// per chunk on the production path and constructs no extra graph nodes, which
// matters because ggml executes every graph output (§6.8).

struct oaf_stage_sink {
    std::map<std::string, std::vector<float>> stages;
    void put(const std::string& name, const float* p, size_t n) { stages[name].assign(p, p + n); }
};

// Per-chunk conv intermediates, in ggml's [F, T, C] element order — which is
// exactly ONNX's [1, C, T, F] flattened, so no transposition is needed and only
// the time axis has to be trimmed of its halo.
struct oaf_chunk_taps {
    std::vector<float> conv[3];
    int64_t F[3] = {0, 0, 0}; // ne0 per conv output
    int64_t C[3] = {0, 0, 0}; // channel count per conv output
};

// ─── ConvStack, in a ggml graph ─────────────────────────────────────────────

// mel_chunk: T_chunk frames of n_mels floats each, frame-major (so ne0 is the
// mel axis). Returns fc_out floats per frame, same ordering — element (t, f) at
// t * fc_out + f.
static bool oaf_conv_stack_chunk(onsets_and_frames_ctx* ctx, const oaf_conv_stack& cs, const float* mel_chunk,
                                 int T_chunk, std::vector<float>& out, oaf_chunk_taps* taps = nullptr) {
    auto& hp = ctx->hp;
    const int n_mels = (int)hp.n_mels;

    if (ctx->graph_meta.empty())
        ctx->graph_meta.resize(4u * 1024 * 1024);
    ggml_init_params ip = {ctx->graph_meta.size(), ctx->graph_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return false;
    ggml_cgraph* gf = ggml_new_graph(ctx0);

    // [W = n_mels, H = T, C = 1, N = 1]
    ggml_tensor* x = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_mels, T_chunk, 1, 1);
    ggml_set_name(x, "mel");
    ggml_set_input(x);

    ggml_tensor* h = x;
    for (int i = 0; i < 3; i++) {
        h = ggml_conv_2d(ctx0, cs.conv_w[i], h, /*s0*/ 1, /*s1*/ 1, /*p0*/ 1, /*p1*/ 1, /*d0*/ 1, /*d1*/ 1);
        const int64_t OC = cs.conv_w[i]->ne[3];
        h = ggml_add(ctx0, h, ggml_reshape_4d(ctx0, cs.conv_b[i], 1, 1, OC, 1));
        h = ggml_relu(ctx0, h);
        if (i >= 1) {
            // ONNX MaxPool kernel_shape (1, 2) is (time, freq); ggml's k0/s0
            // act on ne0, which is the frequency axis here.
            h = ggml_pool_2d(ctx0, h, GGML_OP_POOL_MAX, /*k0*/ 2, /*k1*/ 1, /*s0*/ 2, /*s1*/ 1, /*p0*/ 0, /*p1*/ 0);
        }
        if (taps) {
            // conv<i> is the ONNX stage after ReLU (i == 0) or after ReLU then
            // MaxPool (i >= 1) — the value names the reference dumper exports.
            const char* nm[3] = {"conv0", "conv1", "conv2"};
            ggml_set_name(h, nm[i]);
            ggml_set_output(h);
            ggml_build_forward_expand(gf, h);
        }
    }
    // h: [F = n_mels/4, T, C = 96, 1]. The reference flattens [B, T, C, F] with
    // C major, so permute time out past the channel axis before making it
    // contiguous: index must become t*(C*F) + c*F + f.
    h = ggml_cont(ctx0, ggml_permute(ctx0, h, 0, 2, 1, 3)); // [F, C, T, 1]
    const int64_t midfeat = h->ne[0] * h->ne[1];
    h = ggml_reshape_2d(ctx0, h, midfeat, T_chunk);
    h = ggml_mul_mat(ctx0, cs.fc_w, h); // [fc_out, T]
    h = ggml_add(ctx0, h, cs.fc_b);
    ggml_set_name(h, "fc_out");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    if (!ctx->alloc_conv)
        ctx->alloc_conv = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->alloc_conv || !ggml_gallocr_alloc_graph(ctx->alloc_conv, gf)) {
        std::fprintf(stderr, "oaf: conv-stack graph allocation failed (T=%d)\n", T_chunk);
        ggml_free(ctx0);
        return false;
    }
    // Re-set every input on every compute: gallocr may hand an input's slot to
    // a later intermediate, and the symptom is chunk 0 exact and chunk 1 onward
    // diverged (playbook §6.6). This assignment is that re-set.
    ggml_tensor* in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(in, mel_chunk, 0, (size_t)n_mels * T_chunk * sizeof(float));

    oaf_set_threads(ctx);
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "oaf: conv-stack graph compute failed\n");
        ggml_free(ctx0);
        return false;
    }
    ggml_tensor* res = ggml_graph_get_tensor(gf, "fc_out");
    out.resize((size_t)ggml_nelements(res));
    ggml_backend_tensor_get(res, out.data(), 0, out.size() * sizeof(float));

    if (taps) {
        const char* nm[3] = {"conv0", "conv1", "conv2"};
        for (int i = 0; i < 3; i++) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, nm[i]);
            if (!t)
                continue;
            taps->F[i] = t->ne[0];
            taps->C[i] = t->ne[2];
            taps->conv[i].resize((size_t)ggml_nelements(t));
            ggml_backend_tensor_get(t, taps->conv[i].data(), 0, taps->conv[i].size() * sizeof(float));
        }
    }

    ggml_free(ctx0);
    return true;
}

// mel: [T, n_mels] row-major. Returns [T, fc_out] row-major.
static bool oaf_conv_stack(onsets_and_frames_ctx* ctx, const oaf_conv_stack& cs, const std::vector<float>& mel, int T,
                           std::vector<float>& out, oaf_stage_sink* sink = nullptr, const char* stage_prefix = "") {
    oaf_bench_stage _b("conv_stack");
    auto& hp = ctx->hp;
    const int n_mels = (int)hp.n_mels;
    const int fc_out = (int)hp.fc_out;
    out.assign((size_t)T * fc_out, 0.0f);

    // Whole-sequence conv taps, assembled from the chunks the same way fc_out
    // is: each chunk contributes only its non-halo interior.
    std::vector<float> tap_all[3];
    int64_t tap_F[3] = {0, 0, 0}, tap_C[3] = {0, 0, 0};

    std::vector<float> chunk_in;
    std::vector<float> chunk_out;
    oaf_chunk_taps taps;
    for (int t0 = 0; t0 < T; t0 += OAF_CONV_CHUNK) {
        const int t1 = std::min(T, t0 + OAF_CONV_CHUNK);
        const int lo = std::max(0, t0 - OAF_CONV_HALO);
        const int hi = std::min(T, t1 + OAF_CONV_HALO);
        const int T_chunk = hi - lo;
        chunk_in.resize((size_t)T_chunk * n_mels);
        std::memcpy(chunk_in.data(), mel.data() + (size_t)lo * n_mels, (size_t)T_chunk * n_mels * sizeof(float));
        if (!oaf_conv_stack_chunk(ctx, cs, chunk_in.data(), T_chunk, chunk_out, sink ? &taps : nullptr))
            return false;
        if ((int)chunk_out.size() != T_chunk * fc_out) {
            std::fprintf(stderr, "oaf: conv stack returned %zu floats, expected %d\n", chunk_out.size(),
                         T_chunk * fc_out);
            return false;
        }
        std::memcpy(out.data() + (size_t)t0 * fc_out, chunk_out.data() + (size_t)(t0 - lo) * fc_out,
                    (size_t)(t1 - t0) * fc_out * sizeof(float));
        if (sink) {
            for (int i = 0; i < 3; i++) {
                if (taps.conv[i].empty())
                    continue;
                const int64_t F = taps.F[i], C = taps.C[i];
                tap_F[i] = F;
                tap_C[i] = C;
                tap_all[i].resize((size_t)C * T * F);
                for (int64_t c = 0; c < C; c++) {
                    std::memcpy(tap_all[i].data() + ((size_t)c * T + t0) * F,
                                taps.conv[i].data() + ((size_t)c * T_chunk + (t0 - lo)) * F,
                                (size_t)(t1 - t0) * F * sizeof(float));
                }
            }
        }
    }
    if (sink) {
        for (int i = 0; i < 3; i++) {
            if (tap_all[i].empty())
                continue;
            sink->put(std::string(stage_prefix) + "conv" + std::to_string(i), tap_all[i].data(), tap_all[i].size());
        }
        sink->put(std::string(stage_prefix) + "fc", out.data(), out.size());
        (void)tap_C;
        (void)tap_F;
    }
    return true;
}

// ─── BiLSTM ─────────────────────────────────────────────────────────────────

// Batched input projection: gates[t] = W · x[t] + b, for every t at once.
// x: [T, in] row-major. Returns [T, 4H] row-major.
static bool oaf_lstm_input_proj(onsets_and_frames_ctx* ctx, const oaf_lstm_dir& d, const float* x, int T, int in_size,
                                int gate_size, std::vector<float>& out) {
    if (ctx->graph_meta.empty())
        ctx->graph_meta.resize(4u * 1024 * 1024);
    ggml_init_params ip = {ctx->graph_meta.size(), ctx->graph_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return false;
    ggml_cgraph* gf = ggml_new_graph(ctx0);

    ggml_tensor* xt = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in_size, T);
    ggml_set_name(xt, "x");
    ggml_set_input(xt);
    ggml_tensor* g = ggml_mul_mat(ctx0, d.W, xt); // [4H, T]
    ggml_set_name(g, "g");
    ggml_set_output(g);
    ggml_build_forward_expand(gf, g);

    if (!ctx->alloc_proj)
        ctx->alloc_proj = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->alloc_proj || !ggml_gallocr_alloc_graph(ctx->alloc_proj, gf)) {
        std::fprintf(stderr, "oaf: LSTM input-projection allocation failed\n");
        ggml_free(ctx0);
        return false;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), x, 0, (size_t)in_size * T * sizeof(float));
    oaf_set_threads(ctx);
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "oaf: LSTM input-projection compute failed\n");
        ggml_free(ctx0);
        return false;
    }
    out.resize((size_t)T * gate_size);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "g"), out.data(), 0, out.size() * sizeof(float));
    ggml_free(ctx0);

    // Fold in the (already Wb + Rb) bias here rather than as a graph node: it
    // is one pass over T×4H and saves an ggml_add of the same size.
    for (int t = 0; t < T; t++) {
        float* row = out.data() + (size_t)t * gate_size;
        for (int g2 = 0; g2 < gate_size; g2++)
            row[g2] += d.b[g2];
    }
    return true;
}

// One direction. gates: [T, 4H] (input projection + bias, already computed).
// Writes h_t into out[t * stride + off .. + H].
static void oaf_lstm_recurrence(const oaf_lstm_dir& d, const std::vector<float>& gates, int T, int H, bool reverse,
                                float* out, int stride, int off) {
    const int gate_size = 4 * H;
    std::vector<float> h(H, 0.0f), c(H, 0.0f), rh(gate_size, 0.0f);
    const float* R = d.R.data();

    for (int step = 0; step < T; step++) {
        const int t = reverse ? (T - 1 - step) : step;
        const float* g = gates.data() + (size_t)t * gate_size;

        if (step == 0) {
            std::fill(rh.begin(), rh.end(), 0.0f);
        } else {
            for (int r = 0; r < gate_size; r++) {
                const float* Rr = R + (size_t)r * H;
                float s = 0.0f;
                for (int i = 0; i < H; i++)
                    s += Rr[i] * h[i];
                rh[r] = s;
            }
        }
        for (int i = 0; i < H; i++) {
            const float it = oaf_sigmoid(g[OAF_GATE_I * H + i] + rh[OAF_GATE_I * H + i]);
            const float ot = oaf_sigmoid(g[OAF_GATE_O * H + i] + rh[OAF_GATE_O * H + i]);
            const float ft = oaf_sigmoid(g[OAF_GATE_F * H + i] + rh[OAF_GATE_F * H + i]);
            const float ct = std::tanh(g[OAF_GATE_C * H + i] + rh[OAF_GATE_C * H + i]);
            c[i] = ft * c[i] + it * ct;
            h[i] = ot * std::tanh(c[i]);
        }
        std::memcpy(out + (size_t)t * stride + off, h.data(), (size_t)H * sizeof(float));
    }
}

// ─── The same recurrence, unrolled into a ggml graph ────────────────────────
//
// Opt in with CRISPASR_OAF_GRAPH_LSTM=1. The scalar path above stays the
// default and is never deleted (dev guide A/B rule 1, playbook §5.7 item 3).
//
// WHY THIS EXISTS, measured rather than assumed. Playbook §8 item 2 lists
// "whether a ggml-graph BiLSTM beats a scalar recurrence at O&F's sequence
// length" as explicitly NOT established, and §1b gives the reason to doubt it:
// core/lstm.h unrolls ~12 nodes per timestep, and at T ~ 3000 that is a
// ~200k-node graph across three BiLSTMs, where graph-build, tensor overhead and
// gallocr planning all scale linearly and the per-step work is only a 384x1536
// mat-vec. So it was screened before it was written, one direction, this box,
// single-threaded:
//
//   T      scalar (whole cell)    unrolled graph (build + alloc + compute)
//   313    289 ms  0.924 ms/step   2.1 +  4.2 +  54.0 =  60.2 ms  0.192 ms/step
//   1875  1704 ms  0.909 ms/step  10.1 + 30.6 + 321.7 = 362.4 ms  0.193 ms/step
//   3000  2903 ms  0.968 ms/step  13.3 + 48.1 + 517.8 = 579.1 ms  0.193 ms/step
//
// The graph is ~4.7x cheaper and -- the part that answers §1b -- its per-step
// cost is FLAT in T. Node count does not degrade it; build and alloc together
// stay around 10% of the total even at 48000 nodes. The gap is not mysterious:
// docs/improvements/SRC_ISA_GAP.md establishes that everything under src/ ships
// baseline x86-64 with no -march, so the scalar loop above runs SSE2 4-wide at
// ~1.3 GFLOP/s, while ggml's mul_mat reaches its AVX2/FMA kernel through
// runtime dispatch.
//
// NOT BIT-IDENTICAL, and it cannot be: ggml's mul_mat reduces in a different
// order than the serial `s += Rr[i] * h[i]` loop, and this is a feedback path,
// so a last-ulp difference at step 0 propagates. That is exactly why it is
// gated and why the decision rests on the parity harness and the F1 numbers
// rather than on the speedup.
static bool oaf_lstm_graph_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_OAF_GRAPH_LSTM");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

static bool oaf_lstm_recurrence_graph(onsets_and_frames_ctx* ctx, const oaf_lstm_dir& d,
                                      const std::vector<float>& gates, int T, int H, bool reverse, float* out,
                                      int stride, int off) {
    if (!d.R_t)
        return false;
    const int G = 4 * H;

    // Carried across chunks, so chunking is exact rather than approximate.
    std::vector<float> h_carry((size_t)H, 0.0f), c_carry((size_t)H, 0.0f);
    std::vector<float> h_all;

    for (int s0 = 0; s0 < T; s0 += OAF_LSTM_GRAPH_CHUNK) {
        const int s1 = std::min(T, s0 + OAF_LSTM_GRAPH_CHUNK);
        const int n_steps = s1 - s0;

        const size_t n_nodes = (size_t)n_steps * 20 + 64;
        const size_t need = ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false);
        if (ctx->lstm_meta.size() < need)
            ctx->lstm_meta.resize(need);
        ggml_init_params ip = {ctx->lstm_meta.size(), ctx->lstm_meta.data(), true};
        ggml_context* c0 = ggml_init(ip);
        if (!c0)
            return false;
        ggml_cgraph* gf = ggml_new_graph_custom(c0, n_nodes, false);

        // Inputs. EVERY input is re-set after the alloc below, including the
        // "constant-shaped" carries: gallocr may hand an input's slot to a
        // later intermediate and the symptom is chunk 0 exact, chunk 1 onward
        // diverged (playbook §6.6).
        ggml_tensor* g_in = ggml_new_tensor_2d(c0, GGML_TYPE_F32, G, n_steps);
        ggml_set_name(g_in, "gates");
        ggml_set_input(g_in);
        ggml_tensor* h0 = ggml_new_tensor_1d(c0, GGML_TYPE_F32, H);
        ggml_set_name(h0, "h0");
        ggml_set_input(h0);
        ggml_tensor* c0t = ggml_new_tensor_1d(c0, GGML_TYPE_F32, H);
        ggml_set_name(c0t, "c0");
        ggml_set_input(c0t);
        ggml_tensor* h_out = ggml_new_tensor_2d(c0, GGML_TYPE_F32, H, n_steps);
        ggml_set_name(h_out, "h_all");
        ggml_set_output(h_out);

        ggml_tensor* h = h0;
        ggml_tensor* cs = c0t;
        for (int k = 0; k < n_steps; k++) {
            ggml_tensor* g = ggml_view_1d(c0, g_in, G, (size_t)k * G * sizeof(float));
            // The one piece of real arithmetic per step: R @ h_{t-1}.
            ggml_tensor* sum = ggml_add(c0, g, ggml_mul_mat(c0, d.R_t, h));
            // ONNX gate order is i, o, f, c — not PyTorch's i, f, g, o.
            ggml_tensor* it = ggml_sigmoid(c0, ggml_view_1d(c0, sum, H, (size_t)OAF_GATE_I * H * sizeof(float)));
            ggml_tensor* ot = ggml_sigmoid(c0, ggml_view_1d(c0, sum, H, (size_t)OAF_GATE_O * H * sizeof(float)));
            ggml_tensor* ft = ggml_sigmoid(c0, ggml_view_1d(c0, sum, H, (size_t)OAF_GATE_F * H * sizeof(float)));
            ggml_tensor* ct = ggml_tanh(c0, ggml_view_1d(c0, sum, H, (size_t)OAF_GATE_C * H * sizeof(float)));
            cs = ggml_add(c0, ggml_mul(c0, ft, cs), ggml_mul(c0, it, ct));
            h = ggml_mul(c0, ot, ggml_tanh(c0, cs));
            ggml_build_forward_expand(gf, ggml_cpy(c0, h, ggml_view_1d(c0, h_out, H, (size_t)k * H * sizeof(float))));
        }
        // The final (h, c) are read back to seed the next chunk, so they must
        // survive allocation too.
        ggml_set_name(cs, "c_last");
        ggml_set_output(cs);
        ggml_build_forward_expand(gf, cs);
        ggml_tensor* h_last = h;
        ggml_set_name(h_last, "h_last");
        ggml_set_output(h_last);
        ggml_build_forward_expand(gf, h_last);

        if (!ctx->alloc_lstm)
            ctx->alloc_lstm = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ctx->alloc_lstm || !ggml_gallocr_alloc_graph(ctx->alloc_lstm, gf)) {
            std::fprintf(stderr, "oaf: graph BiLSTM allocation failed (steps=%d, nodes=%d)\n", n_steps,
                         ggml_graph_n_nodes(gf));
            ggml_free(c0);
            return false;
        }

        // Gates for this chunk, in step order — reversed for the backward pass.
        std::vector<float> g_chunk((size_t)n_steps * G);
        for (int k = 0; k < n_steps; k++) {
            const int t = reverse ? (T - 1 - (s0 + k)) : (s0 + k);
            std::memcpy(g_chunk.data() + (size_t)k * G, gates.data() + (size_t)t * G, (size_t)G * sizeof(float));
        }
        ggml_backend_tensor_set(g_in, g_chunk.data(), 0, g_chunk.size() * sizeof(float));
        ggml_backend_tensor_set(h0, h_carry.data(), 0, (size_t)H * sizeof(float));
        ggml_backend_tensor_set(c0t, c_carry.data(), 0, (size_t)H * sizeof(float));

        oaf_set_threads(ctx);
        if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "oaf: graph BiLSTM compute failed\n");
            ggml_free(c0);
            return false;
        }

        h_all.resize((size_t)n_steps * H);
        ggml_backend_tensor_get(h_out, h_all.data(), 0, h_all.size() * sizeof(float));
        ggml_backend_tensor_get(h_last, h_carry.data(), 0, (size_t)H * sizeof(float));
        ggml_backend_tensor_get(cs, c_carry.data(), 0, (size_t)H * sizeof(float));
        for (int k = 0; k < n_steps; k++) {
            const int t = reverse ? (T - 1 - (s0 + k)) : (s0 + k);
            std::memcpy(out + (size_t)t * stride + off, h_all.data() + (size_t)k * H, (size_t)H * sizeof(float));
        }
        ggml_free(c0);
    }
    return true;
}

// Dispatch. The scalar path is the default; the graph path is opt-in and falls
// BACK to the scalar one if it fails, so the gate can never make the model stop
// working — only slower.
static bool oaf_run_recurrence(onsets_and_frames_ctx* ctx, const oaf_lstm_dir& d, const std::vector<float>& gates,
                               int T, int H, bool reverse, float* out, int stride, int off) {
    if (oaf_lstm_graph_enabled()) {
        if (oaf_lstm_recurrence_graph(ctx, d, gates, T, H, reverse, out, stride, off))
            return true;
        std::fprintf(stderr, "oaf: graph BiLSTM failed, falling back to the scalar recurrence\n");
    }
    oaf_lstm_recurrence(d, gates, T, H, reverse, out, stride, off);
    return true;
}

// x: [T, in] row-major. Returns [T, 2H] row-major, forward half first —
// the order the ONNX graph's Transpose/Reshape pair produces.
static bool oaf_bilstm(onsets_and_frames_ctx* ctx, const oaf_lstm& l, const std::vector<float>& x, int T,
                       std::vector<float>& out, oaf_stage_sink* sink = nullptr, const char* stage_name = "") {
    oaf_bench_stage _b("bilstm");
    const int H = (int)ctx->hp.lstm_hidden;
    const int in_size = l.input_size;
    const int gate_size = 4 * H;
    out.assign((size_t)T * 2 * H, 0.0f);

    std::vector<float> gates;
    if (!oaf_lstm_input_proj(ctx, l.fwd, x.data(), T, in_size, gate_size, gates))
        return false;
    if (!oaf_run_recurrence(ctx, l.fwd, gates, T, H, /*reverse*/ false, out.data(), 2 * H, 0))
        return false;

    if (!oaf_lstm_input_proj(ctx, l.rev, x.data(), T, in_size, gate_size, gates))
        return false;
    if (!oaf_run_recurrence(ctx, l.rev, gates, T, H, /*reverse*/ true, out.data(), 2 * H, H))
        return false;
    if (sink)
        sink->put(stage_name, out.data(), out.size());
    return true;
}

// ─── Head ───────────────────────────────────────────────────────────────────

// x: [T, in] row-major. Returns [T, 88] LOGITS, row-major.
static bool oaf_apply_head(onsets_and_frames_ctx* ctx, const oaf_head& hd, const std::vector<float>& x, int T,
                           int in_size, std::vector<float>& out, oaf_stage_sink* sink = nullptr,
                           const char* stage_name = "") {
    if (ctx->graph_meta.empty())
        ctx->graph_meta.resize(4u * 1024 * 1024);
    ggml_init_params ip = {ctx->graph_meta.size(), ctx->graph_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return false;
    ggml_cgraph* gf = ggml_new_graph(ctx0);

    ggml_tensor* xt = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in_size, T);
    ggml_set_name(xt, "x");
    ggml_set_input(xt);
    ggml_tensor* y = ggml_add(ctx0, ggml_mul_mat(ctx0, hd.w, xt), hd.b);
    ggml_set_name(y, "y");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    if (!ctx->alloc_head)
        ctx->alloc_head = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->alloc_head || !ggml_gallocr_alloc_graph(ctx->alloc_head, gf)) {
        ggml_free(ctx0);
        return false;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), x.data(), 0, (size_t)in_size * T * sizeof(float));
    oaf_set_threads(ctx);
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return false;
    }
    ggml_tensor* res = ggml_graph_get_tensor(gf, "y");
    out.resize((size_t)ggml_nelements(res));
    ggml_backend_tensor_get(res, out.data(), 0, out.size() * sizeof(float));
    ggml_free(ctx0);
    if (sink)
        sink->put(stage_name, out.data(), out.size());
    return true;
}

// ─── One forward pass ───────────────────────────────────────────────────────

struct oaf_heads {
    std::vector<float> onset, offset, activation, frame, velocity; // [T, 88] sigmoid-ed
    int T = 0;
};

static bool oaf_forward(onsets_and_frames_ctx* ctx, const float* pcm, int n_samples, oaf_heads& out,
                        oaf_stage_sink* sink = nullptr, const std::vector<float>* mel_override = nullptr) {
    auto& hp = ctx->hp;
    auto& w = ctx->weights;
    const int K = (int)hp.classes_num;
    const int fc_out = (int)hp.fc_out;
    const int H2 = 2 * (int)hp.lstm_hidden;

    int T = 0;
    std::vector<float> mel;
    if (mel_override) {
        mel = *mel_override;
        T = (int)(mel.size() / (size_t)hp.n_mels);
    } else {
        mel = oaf_log_mel(ctx, pcm, n_samples, T);
    }
    if (T <= 0)
        return false;
    out.T = T;
    if (sink)
        sink->put("mel", mel.data(), mel.size());

    std::vector<float> cs_out, lstm_out;

    // onset
    if (!oaf_conv_stack(ctx, w.onset_cs, mel, T, cs_out, sink, "onset_"))
        return false;
    if (!oaf_bilstm(ctx, w.onset_lstm, cs_out, T, lstm_out, sink, "onset_bilstm"))
        return false;
    if (!oaf_apply_head(ctx, w.onset_head, lstm_out, T, H2, out.onset, sink, "onset_logits"))
        return false;

    // offset
    if (!oaf_conv_stack(ctx, w.offset_cs, mel, T, cs_out, sink, "offset_"))
        return false;
    if (!oaf_bilstm(ctx, w.offset_lstm, cs_out, T, lstm_out, sink, "offset_bilstm"))
        return false;
    if (!oaf_apply_head(ctx, w.offset_head, lstm_out, T, H2, out.offset, sink, "offset_logits"))
        return false;

    // activation (frame_stack — NOT the frame head, whatever the export calls it)
    if (!oaf_conv_stack(ctx, w.activation_cs, mel, T, cs_out, sink, "activation_"))
        return false;
    if (!oaf_apply_head(ctx, w.activation_head, cs_out, T, fc_out, out.activation, sink, "activation_logits"))
        return false;

    // velocity
    if (!oaf_conv_stack(ctx, w.velocity_cs, mel, T, cs_out, sink, "velocity_"))
        return false;
    if (!oaf_apply_head(ctx, w.velocity_head, cs_out, T, fc_out, out.velocity, sink, "velocity_logits"))
        return false;

    // combined stack: cat(onset, offset, activation) — LOGITS, no sigmoid,
    // exactly as the graph's Concat takes the three Add outputs.
    std::vector<float> comb((size_t)T * 3 * K);
    for (int t = 0; t < T; t++) {
        float* row = comb.data() + (size_t)t * 3 * K;
        std::memcpy(row, out.onset.data() + (size_t)t * K, (size_t)K * sizeof(float));
        std::memcpy(row + K, out.offset.data() + (size_t)t * K, (size_t)K * sizeof(float));
        std::memcpy(row + 2 * K, out.activation.data() + (size_t)t * K, (size_t)K * sizeof(float));
    }
    if (sink)
        sink->put("combined_input", comb.data(), comb.size());
    if (!oaf_bilstm(ctx, w.frame_lstm, comb, T, lstm_out, sink, "frame_bilstm"))
        return false;
    if (!oaf_apply_head(ctx, w.frame_head, lstm_out, T, H2, out.frame, sink, "frame_logits"))
        return false;

    for (auto* v : {&out.onset, &out.offset, &out.activation, &out.frame, &out.velocity})
        for (auto& x : *v)
            x = oaf_sigmoid(x);
    return true;
}

// ─── Note decoding (modules/decoding.py: extract_notes) ─────────────────────

static void oaf_extract_notes(const oaf_heads& h, const oaf_hparams& hp, float onset_threshold, float frame_threshold,
                              std::vector<onsets_and_frames_note_event>& notes) {
    const int K = (int)hp.classes_num;
    const int T = h.T;
    const float scale = (float)hp.hop_size / (float)hp.sample_rate;

    for (int p = 0; p < K; p++) {
        for (int t = 0; t < T; t++) {
            const bool on = h.onset[(size_t)t * K + p] > onset_threshold;
            const bool prev = t > 0 && h.onset[(size_t)(t - 1) * K + p] > onset_threshold;
            if (!on || prev)
                continue;
            int off = t;
            float vel_sum = 0.0f;
            int vel_n = 0;
            while (off < T &&
                   (h.onset[(size_t)off * K + p] > onset_threshold || h.frame[(size_t)off * K + p] > frame_threshold)) {
                vel_sum += h.velocity[(size_t)off * K + p];
                vel_n++;
                off++;
            }
            if (off <= t)
                continue;
            int v = (int)std::lround(127.0f * (vel_n ? vel_sum / (float)vel_n : 0.0f));
            v = std::max(0, std::min(127, v));
            notes.push_back({(float)t * scale, (float)off * scale, (int)hp.begin_note + p, v});
        }
    }
    std::sort(notes.begin(), notes.end(),
              [](const onsets_and_frames_note_event& a, const onsets_and_frames_note_event& b) {
                  return a.onset_time < b.onset_time;
              });
}

// ─── Public entry point ─────────────────────────────────────────────────────

int onsets_and_frames_transcribe(struct onsets_and_frames_ctx* ctx, const float* pcm, int n_samples,
                                 struct onsets_and_frames_result* result) {
    if (!ctx || !pcm || n_samples <= 0 || !result)
        return -1;
    std::memset(result, 0, sizeof(*result));
    auto& hp = ctx->hp;
    auto& p = ctx->params;
    const int K = (int)hp.classes_num;

    const int seg_samples = p.segment_seconds > 0.0f ? (int)(p.segment_seconds * (float)hp.sample_rate) : 0;

    oaf_heads all;
    if (seg_samples <= 0 || n_samples <= seg_samples) {
        if (!oaf_forward(ctx, pcm, n_samples, all))
            return -2;
    } else {
        // 50% overlap, keep the middle half of each segment. The LSTM is the
        // only part with memory across frames, so the discarded margins are
        // exactly the frames whose state has not warmed up.
        // Each segment contributes exactly its middle half, which is
        // hop_samples/hop_size frames — the same stride the pointer advances
        // by — so appending them in order reconstructs a dense timeline with
        // no gap and no overlap.
        const int hop_samples = seg_samples / 2;
        int seg_index = 0;
        int next_global = 0; // first global frame not yet written
        for (int ptr = 0; ptr < n_samples; ptr += hop_samples, seg_index++) {
            const int len = std::min(seg_samples, n_samples - ptr);
            if (len < (int)hp.n_fft)
                break;
            oaf_heads seg;
            if (!oaf_forward(ctx, pcm + ptr, len, seg))
                return -2;
            const bool last = (ptr + hop_samples >= n_samples);
            // Derive the seam from the GLOBAL frame clock rather than from
            // seg.T/4: the final segment is usually short, so a quarter of its
            // own length is the wrong margin and would leave a gap.
            const int seg_first_frame = ptr / (int)hp.hop_size;
            int start = std::max(seg_index == 0 ? 0 : seg.T / 4, next_global - seg_first_frame);
            const int end = last ? seg.T : (3 * seg.T) / 4;
            if (start > end)
                start = end;
            const std::pair<std::vector<float>*, const std::vector<float>*> parts[] = {
                {&all.onset, &seg.onset}, {&all.offset, &seg.offset},     {&all.activation, &seg.activation},
                {&all.frame, &seg.frame}, {&all.velocity, &seg.velocity},
            };
            for (const auto& pr : parts) {
                pr.first->insert(pr.first->end(), pr.second->begin() + (size_t)start * K,
                                 pr.second->begin() + (size_t)end * K);
            }
            all.T += (end - start);
            next_global = seg_first_frame + end;
            if (last)
                break;
        }
        if (all.T <= 0)
            return -2;
    }

    std::vector<onsets_and_frames_note_event> notes;
    oaf_extract_notes(all, hp, p.onset_threshold, p.frame_threshold, notes);

    result->n_notes = (int)notes.size();
    result->n_frames = all.T;
    result->n_classes = K;
    if (!notes.empty()) {
        result->note_events =
            (onsets_and_frames_note_event*)std::malloc(notes.size() * sizeof(onsets_and_frames_note_event));
        if (!result->note_events)
            return -3;
        std::memcpy(result->note_events, notes.data(), notes.size() * sizeof(onsets_and_frames_note_event));
    }

    if (p.verbosity >= 2) {
        auto dup = [&](const std::vector<float>& v) -> float* {
            float* b = (float*)std::malloc(v.size() * sizeof(float));
            if (b)
                std::memcpy(b, v.data(), v.size() * sizeof(float));
            return b;
        };
        result->onset_output = dup(all.onset);
        result->offset_output = dup(all.offset);
        result->frame_output = dup(all.frame);
        result->activation_output = dup(all.activation);
        result->velocity_output = dup(all.velocity);
    }
    return 0;
}

void onsets_and_frames_result_free(struct onsets_and_frames_result* result) {
    if (!result)
        return;
    std::free(result->note_events);
    std::free(result->onset_output);
    std::free(result->offset_output);
    std::free(result->frame_output);
    std::free(result->activation_output);
    std::free(result->velocity_output);
    std::memset(result, 0, sizeof(*result));
}

// ─── Diff harness ───────────────────────────────────────────────────────────
//
// Per-stage cosine parity against the ONNX reference, on the pattern
// src/basic_pitch.cpp:824 established. Registered as
// `crispasr-diff onsets-and-frames <model.gguf> <ref.gguf> <audio.wav>`.
//
// The reference GGUF is produced by tools/reference_backends/onsets_and_frames.py,
// which runs the ONNX graph under onnxruntime with the intermediate value names
// promoted to graph outputs. That script is fed the mel THIS runtime computed
// (oaf-parity-dump writes it), so downstream stages isolate the model from the
// front end exactly as tools/oaf_parity.py:5-11 does — and the `mel` stage is
// still compared first, so a front-end difference surfaces as itself rather
// than silently shifting every cosine after it.
//
// COSINE IS SCALE-BLIND. The |mine| / |ref| RMS columns are printed for every
// stage and are the thing to read when a stage "passes" but the model is wrong
// by a uniform factor (playbook §5.0, HARD RULE 2b).

static double oaf_cosine(const float* a, const float* b, int64_t n) {
    double num = 0.0, na = 0.0, nb = 0.0;
    for (int64_t i = 0; i < n; i++) {
        num += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0)
        return 0.0;
    return num / (std::sqrt(na) * std::sqrt(nb));
}

static double oaf_rms(const float* a, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++)
        s += (double)a[i] * (double)a[i];
    return n > 0 ? std::sqrt(s / (double)n) : 0.0;
}

static bool oaf_ref_get(const core_gguf::WeightLoad& rw, const std::string& name, std::vector<float>& out) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end())
        return false;
    const int64_t n = ggml_nelements(it->second);
    out.resize((size_t)n);
    ggml_backend_tensor_get(it->second, out.data(), 0, (size_t)n * sizeof(float));
    return true;
}

int onsets_and_frames_diff(const char* model_gguf, const char* ref_gguf, const float* pcm_16k, int n_samples,
                           int verbosity) {
    onsets_and_frames_params p = onsets_and_frames_default_params();
    p.verbosity = 0;
    p.n_threads = 1; // determinism first; this harness is not a timing run
    onsets_and_frames_ctx* ctx = onsets_and_frames_init_from_file(model_gguf, p);
    if (!ctx) {
        std::fprintf(stderr, "onsets_and_frames_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "oaf_ref", rw)) {
        std::fprintf(stderr, "onsets_and_frames_diff: failed to load reference %s\n", ref_gguf);
        onsets_and_frames_free(ctx);
        return 2;
    }

    // The reference was dumped on a mel this same runtime produced, so replay
    // that mel rather than recomputing it — then the model stages compare the
    // model alone. The `mel` stage below still checks our recomputation of it.
    std::vector<float> ref_mel;
    const bool have_ref_mel = oaf_ref_get(rw, "mel", ref_mel);

    oaf_stage_sink sink;
    oaf_heads heads;
    if (!oaf_forward(ctx, pcm_16k, n_samples, heads, &sink, have_ref_mel ? &ref_mel : nullptr)) {
        std::fprintf(stderr, "onsets_and_frames_diff: forward pass failed\n");
        core_gguf::free_weights(rw);
        onsets_and_frames_free(ctx);
        return 2;
    }

    // Our own front end, compared against the reference's copy of it. With the
    // replay above this is the ONLY place a mel difference can show, which is
    // the point: it cannot hide inside a downstream cosine.
    if (have_ref_mel) {
        int T_mine = 0;
        auto mine_mel = oaf_log_mel(ctx, pcm_16k, n_samples, T_mine);
        sink.stages["mel"] = mine_mel;
    }

    // The global gate. examples/cli/crispasr_diff.h:101 uses the same value.
    const double COS_THRESHOLD = 0.999;
    int n_fail = 0, n_cmp = 0;

    static const char* kStages[] = {
        "mel",           "onset_conv0",       "onset_conv1",      "onset_conv2",
        "onset_fc",      "offset_conv0",      "offset_conv1",     "offset_conv2",
        "offset_fc",     "activation_conv0",  "activation_conv1", "activation_conv2",
        "activation_fc", "velocity_conv0",    "velocity_conv1",   "velocity_conv2",
        "velocity_fc",   "onset_bilstm",      "onset_logits",     "offset_bilstm",
        "offset_logits", "activation_logits", "velocity_logits",  "combined_input",
        "frame_bilstm",  "frame_logits",
    };

    std::fprintf(stderr, "onsets-and-frames diff (T=%d, n_samples=%d, mel %s):\n", heads.T, n_samples,
                 have_ref_mel ? "replayed from reference" : "recomputed");
    for (const char* stage : kStages) {
        auto it = sink.stages.find(stage);
        std::vector<float> ref;
        if (it == sink.stages.end() || !oaf_ref_get(rw, stage, ref)) {
            if (verbosity >= 2)
                std::fprintf(stderr, "  %-20s SKIP (absent from %s)\n", stage,
                             it == sink.stages.end() ? "runtime" : "reference");
            continue;
        }
        const std::vector<float>& mine = it->second;
        n_cmp++;
        const int64_t n = (int64_t)std::min(mine.size(), ref.size());
        const double cos = oaf_cosine(mine.data(), ref.data(), n);
        double mad = 0.0;
        for (int64_t i = 0; i < n; i++)
            mad = std::max(mad, std::fabs((double)mine[i] - (double)ref[i]));
        const bool ok = cos >= COS_THRESHOLD && mine.size() == ref.size();
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok) {
            std::fprintf(stderr, "  %-20s %s cos=%.7f max_abs=%.3e  |mine|=%.6g |ref|=%.6g  (n=%zu/%zu)\n", stage,
                         ok ? "PASS" : "FAIL", cos, mad, oaf_rms(mine.data(), n), oaf_rms(ref.data(), n), mine.size(),
                         ref.size());
        }
    }

    core_gguf::free_weights(rw);
    onsets_and_frames_free(ctx);
    std::fprintf(stderr, "onsets-and-frames diff: %s (%d of %d stages failing)\n", n_fail == 0 ? "PASS" : "FAIL",
                 n_fail, n_cmp);
    if (n_cmp == 0) {
        std::fprintf(stderr, "onsets-and-frames diff: no stage was compared — the reference is empty or misnamed\n");
        return 2;
    }
    return n_fail == 0 ? 0 : 1;
}
