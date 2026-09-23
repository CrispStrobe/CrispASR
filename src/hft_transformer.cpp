// hft_transformer.cpp — hFT-Transformer piano transcription backend.
//
// See hft_transformer.h for the architecture, the fused front end and why the
// velocity gate is load-bearing. Conventions follow src/onsets_and_frames.cpp,
// which is the piano arm this one is measured against.

#include "hft_transformer.h"

#include "core/crispasr_env.h"
#include "core/fft.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
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
#include <string>
#include <thread>
#include <vector>

// ─── Hyperparameters ────────────────────────────────────────────────────────

struct hft_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_fft = 2048;
    uint32_t hop_size = 256;
    uint32_t n_mels = 256;
    float fmin = 0.0f;
    float fmax = 8000.0f;
    float mel_eps = 1e-8f;
    uint32_t n_frame = 128; // frames answered per window
    uint32_t n_margin = 32; // margin frames each side
    uint32_t classes_num = 88;
    uint32_t begin_note = 21; // MIDI A0
    uint32_t hidden = 256;
    uint32_t n_heads = 4;
    uint32_t pf_dim = 512;
    uint32_t velocity_bins = 128;
    uint32_t enc_layers = 3;
    uint32_t dec_freq_layers = 3; // layer_zero + 2
    uint32_t dec_time_layers = 3;
    float ln_eps = 1e-5f;
    uint32_t front_taps = 65;
};

// Frames per encoder / decoder-freq chunk. Nothing in either stage mixes
// across frames, so a chunk is bit-identical to the unchunked result; this
// only bounds the attention score tensor, which at [256, 256, 4, chunk] is
// by far the largest thing the graph allocates.
static constexpr int HFT_FRAME_CHUNK_DEFAULT = 32;

// ─── Weights ────────────────────────────────────────────────────────────────

struct hft_linear {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct hft_attn {
    hft_linear q, k, v, o;
};

// One transformer layer. `self` and `cross` are both present only in the
// middle frequency-decoder layers; `layer_zero_freq` has cross only and the
// encoder / time decoder have self only. A single LayerNorm module is applied
// after every residual add — the checkpoint has one set of gains per layer.
struct hft_layer {
    bool has_self = false;
    bool has_cross = false;
    hft_attn self, cross;
    hft_linear ff1, ff2;
    ggml_tensor* ln_w = nullptr;
    ggml_tensor* ln_b = nullptr;
};

struct hft_weights {
    hft_linear front;                    // fused Conv2d(1,4,(1,5)) + Linear(244, 256)
    ggml_tensor* pos_enc_freq = nullptr; // [256 dim, 256 pos] in ggml ne order
    ggml_tensor* pos_dec_freq = nullptr; // [256 dim,  88 pos]
    ggml_tensor* pos_dec_time = nullptr; // [256 dim, 128 pos]
    std::vector<hft_layer> enc, dec_freq, dec_time;
    hft_linear head_onset, head_offset, head_mpe, head_velocity;
    ggml_tensor* mel_fb = nullptr;
    ggml_tensor* window = nullptr;
};

struct hft_transformer_ctx {
    hft_hparams hp;
    hft_weights weights;
    hft_transformer_params params;

    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    // load_weights_repack() returns TWO weight buffers: the repack partition
    // in wl.buf and the default one in wl.buf_cpu. Both must be released, and
    // wl.buf_cpu is null on the ordinary (non-repack) path.
    ggml_backend_buffer_t w_buf_cpu = nullptr;
    ggml_backend_t backend = nullptr;

    std::vector<float> mel_fb; // [n_mels * n_freqs], MelsFreqs layout
    std::vector<float> hann;   // [n_fft]
    std::vector<uint8_t> graph_meta;

    // One allocator per graph shape, kept for the life of the context.
    // §36.4 of the flutter_tuner benchmark named "a fresh ggml allocator per
    // convolution chunk" as one of the two unfixed causes of the Onsets &
    // Frames arm's throughput gap; reusing them costs nothing (ggml_gallocr
    // keeps its buffer when the next graph fits) and this model runs four
    // encoder chunks and a time decoder for every 2.048 s of audio.
    ggml_gallocr_t alloc_enc = nullptr;
    ggml_gallocr_t alloc_time = nullptr;
};

// ─── Bench instrumentation (docs/contributing.md §1) ────────────────────────

static bool hft_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("HFT_TRANSFORMER_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct hft_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit hft_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~hft_bench_stage() {
        if (!hft_bench_enabled())
            return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  hft_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ─── Small helpers ──────────────────────────────────────────────────────────

static void hft_fft_r2c(const float* in, int N, float* out) {
    std::vector<float> re(N), im(N, 0.0f);
    std::memcpy(re.data(), in, N * sizeof(float));
    core_fft::fft_radix2_inplace(re.data(), im.data(), N);
    for (int i = 0; i < N; i++) {
        out[2 * i + 0] = re[i];
        out[2 * i + 1] = im[i];
    }
}

// Read a weight tensor to host F32.
//
// The raw `t->data` pointer is only dereferenceable when the tensor lives in a
// host buffer. That was always true while this file hard-coded the CPU
// backend; it is NOT true once the weights can land in a Metal/CUDA/Vulkan
// buffer, where `data` is a device address (or, on Metal, an offset into an
// MTLBuffer that happens to be readable only because Apple Silicon is unified
// — relying on that is how a CUDA build gets a silent segfault). So stage the
// bytes through ggml_backend_tensor_get(), which every backend implements, and
// dequantise from the staging copy.
static std::vector<float> hft_to_f32(const ggml_tensor* t) {
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
    if (tr && tr->to_float)
        tr->to_float(src, out.data(), n);
    else
        std::fprintf(stderr, "hft: no dequantiser for tensor type %s\n", ggml_type_name(t->type));
    return out;
}

static int hft_nthreads(const hft_transformer_ctx* ctx) {
    if (ctx && ctx->params.n_threads > 0)
        return ctx->params.n_threads;
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : (int)std::min(hw, 8u);
}

// core_cpu_backend::set_n_threads() is ggml_backend_cpu_set_n_threads() in the
// ordinary (non-DL) build, and that one asserts ggml_backend_is_cpu(). Calling
// it on a Metal backend aborts the process, so every site is guarded now that
// ctx->backend may not be the CPU.
static inline void hft_set_threads(const hft_transformer_ctx* ctx) {
    if (core_cpu_backend::is_cpu(ctx->backend))
        core_cpu_backend::set_n_threads(ctx->backend, hft_nthreads(ctx));
}

static inline float hft_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ─── Weight loading ─────────────────────────────────────────────────────────

static bool load_linear(core_gguf::tensor_map& tm, const std::string& p, hft_linear& l) {
    l.w = core_gguf::require(tm, (p + ".weight").c_str(), "hft");
    l.b = core_gguf::require(tm, (p + ".bias").c_str(), "hft");
    return l.w && l.b;
}

static bool load_attn(core_gguf::tensor_map& tm, const std::string& p, hft_attn& a) {
    return load_linear(tm, p + ".q", a.q) && load_linear(tm, p + ".k", a.k) && load_linear(tm, p + ".v", a.v) &&
           load_linear(tm, p + ".o", a.o);
}

static bool load_layer(core_gguf::tensor_map& tm, const std::string& p, hft_layer& l, bool self, bool cross) {
    l.has_self = self;
    l.has_cross = cross;
    if (self && !load_attn(tm, p + ".attn", l.self))
        return false;
    if (cross && !load_attn(tm, p + ".xattn", l.cross))
        return false;
    if (!load_linear(tm, p + ".ff1", l.ff1) || !load_linear(tm, p + ".ff2", l.ff2))
        return false;
    l.ln_w = core_gguf::require(tm, (p + ".ln.weight").c_str(), "hft");
    l.ln_b = core_gguf::require(tm, (p + ".ln.bias").c_str(), "hft");
    return l.ln_w && l.ln_b;
}

struct hft_transformer_params hft_transformer_default_params(void) {
    return {
        /* n_threads            */ 4,
        /* verbosity            */ 1,
        /* use_gpu              */ false,
        /* onset_threshold      */ 0.5f,
        /* offset_threshold     */ 0.5f,
        /* mpe_threshold        */ 0.5f,
        /* ignore_zero_velocity */ true,
        /* frame_chunk          */ 0,
    };
}

struct hft_transformer_ctx* hft_transformer_init_from_file(const char* path, struct hft_transformer_params params) {
    auto* ctx = new hft_transformer_ctx();
    ctx->params = params;

    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete ctx;
        return nullptr;
    }
    auto& hp = ctx->hp;
    hp.sample_rate = core_gguf::kv_u32(meta, "hft.sample_rate", hp.sample_rate);
    hp.n_fft = core_gguf::kv_u32(meta, "hft.n_fft", hp.n_fft);
    hp.hop_size = core_gguf::kv_u32(meta, "hft.hop_size", hp.hop_size);
    hp.n_mels = core_gguf::kv_u32(meta, "hft.n_mels", hp.n_mels);
    hp.fmin = core_gguf::kv_f32(meta, "hft.fmin", hp.fmin);
    hp.fmax = core_gguf::kv_f32(meta, "hft.fmax", hp.fmax);
    hp.mel_eps = core_gguf::kv_f32(meta, "hft.mel_eps", hp.mel_eps);
    hp.n_frame = core_gguf::kv_u32(meta, "hft.n_frame", hp.n_frame);
    hp.n_margin = core_gguf::kv_u32(meta, "hft.n_margin", hp.n_margin);
    hp.classes_num = core_gguf::kv_u32(meta, "hft.classes_num", hp.classes_num);
    hp.begin_note = core_gguf::kv_u32(meta, "hft.begin_note", hp.begin_note);
    hp.hidden = core_gguf::kv_u32(meta, "hft.hidden", hp.hidden);
    hp.n_heads = core_gguf::kv_u32(meta, "hft.n_heads", hp.n_heads);
    hp.pf_dim = core_gguf::kv_u32(meta, "hft.pf_dim", hp.pf_dim);
    hp.velocity_bins = core_gguf::kv_u32(meta, "hft.velocity_bins", hp.velocity_bins);
    hp.enc_layers = core_gguf::kv_u32(meta, "hft.enc_layers", hp.enc_layers);
    hp.dec_freq_layers = core_gguf::kv_u32(meta, "hft.dec_freq_layers", hp.dec_freq_layers);
    hp.dec_time_layers = core_gguf::kv_u32(meta, "hft.dec_time_layers", hp.dec_time_layers);
    hp.ln_eps = core_gguf::kv_f32(meta, "hft.ln_eps", hp.ln_eps);
    hp.front_taps = core_gguf::kv_u32(meta, "hft.front_taps", hp.front_taps);

    // The converter folds the (1,5) convolution into the token embedding. A
    // GGUF that says otherwise carries a 244-wide embedding this runtime has
    // no conv to feed, and would run on garbage rather than fail.
    const std::string front = core_gguf::kv_str(meta, "hft.front_end", "fused-conv-tok-embedding");

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
    if (front != "fused-conv-tok-embedding") {
        std::fprintf(stderr, "hft: GGUF declares front end '%s'; this runtime implements the fused one\n",
                     front.c_str());
        delete ctx;
        return nullptr;
    }
    if (hp.front_taps != 2 * hp.n_margin + 1) {
        std::fprintf(stderr, "hft: front_taps %u does not match 2*n_margin+1 = %u\n", hp.front_taps,
                     2 * hp.n_margin + 1);
        delete ctx;
        return nullptr;
    }
    if (hp.hidden % hp.n_heads != 0) {
        std::fprintf(stderr, "hft: hidden %u is not divisible by n_heads %u\n", hp.hidden, hp.n_heads);
        delete ctx;
        return nullptr;
    }

    // CUDA > Metal > Vulkan > CPU, the src/crepe.cpp:346 pattern (issue #214).
    // hFT is 83.5% dense weight GEMM by FLOP (§36.2 of the flutter_tuner
    // benchmark: 248.6 GFLOP per 2.048 s window, only 16.5% attention), so it
    // is the piano arm with the most to gain from a GPU — dense matmul is what
    // Metal is for.
    //
    // Two knobs, and they mean different things on purpose:
    //   * params.use_gpu — the caller's intent. It was already in the struct
    //     and was never read; that is now fixed rather than deleted. The CLI
    //     sets it from --no-gpu / --gpu-backend (whisper_params::use_gpu
    //     defaults true), the library default stays false.
    //   * CRISPASR_HFT_NO_GPU=1 — forces CPU whatever the caller asked for, so
    //     an A/B can run both arms from one binary without a code change.
    const bool no_gpu = crispasr_env::get("CRISPASR_HFT_NO_GPU") != nullptr || !params.use_gpu;
    ctx->backend = no_gpu ? nullptr : crispasr_init_gpu_backend();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    if (!ctx->backend) {
        std::fprintf(stderr, "hft: no ggml backend could be initialised\n");
        delete ctx;
        return nullptr;
    }
    // Not `ctx->backend != nullptr after the GPU call`: crispasr_init_gpu_backend()
    // ends in ggml_backend_init_best(), which returns the CPU backend on a
    // CPU-only build. Ask the device what it is, so the line an A/B greps for
    // cannot claim a GPU that is not there.
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
        std::fprintf(stderr, "hft: backend = %s (%s)\n", ggml_backend_name(ctx->backend),
                     core_cpu_backend::is_cpu(ctx->backend) ? "CPU" : "GPU");

    core_gguf::WeightLoad wl;
    // ggml's repacked int8 GEMM (docs/ggml-optimisation-playbook.md §4) is
    // reached only by putting the weight in the CPU device's extra buffer
    // type. hFT is 83.5% weight GEMM, so it is the model with the most to
    // gain. Every hft_linear::w is used exactly once, as src[0] of
    // ggml_mul_mat in hft_linear_apply(), with an F32 activation — which is
    // precisely the contract load_weights_repack() requires. Nothing else in
    // this model is: the layer-norm weights are ggml_mul operands and the
    // positional tables are ggml_add operands, so both are excluded by name.
    //
    // No-ops (and keeps zero-copy mmap) when the host offers no repack buffer
    // type, or when no weight is of a type it has a kernel for — which is the
    // case for the q8_0 GGUF on x86, where ggml has no q8_0 repack kernel at
    // all. q4_0 and q4_K are the quantisations this helps.
    auto is_hft_matmul_weight = [](const char* name, void*) -> bool {
        const std::string n = name;
        if (n.size() < 7 || n.compare(n.size() - 7, 7, ".weight") != 0)
            return false;
        return n.find(".ln.") == std::string::npos;
    };
    //
    // The repack buffer type is a property of the CPU *device* — ggml offers no
    // such extra buffer type for Metal/CUDA/Vulkan, and load_weights_repack()
    // would classify every weight against a buft that does not exist for this
    // backend. On a GPU backend take the ordinary loader; the GEMM is going to
    // a device kernel anyway, which is the whole point of being there.
    int n_repacked = 0;
    const bool loaded =
        core_cpu_backend::is_cpu(ctx->backend)
            ? core_gguf::load_weights_repack(path, ctx->backend, is_hft_matmul_weight, nullptr, "hft", wl, &n_repacked)
            : core_gguf::load_weights(path, ctx->backend, "hft", wl);
    if (!loaded) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;
    ctx->w_buf_cpu = wl.buf_cpu;

    auto& w = ctx->weights;
    auto& tm = wl.tensors;
    bool ok = load_linear(tm, "hft.encoder.front", w.front);
    w.pos_enc_freq = core_gguf::require(tm, "hft.encoder.pos_freq", "hft");
    w.pos_dec_freq = core_gguf::require(tm, "hft.decoder.pos_freq", "hft");
    w.pos_dec_time = core_gguf::require(tm, "hft.decoder.pos_time", "hft");
    ok = ok && w.pos_enc_freq && w.pos_dec_freq && w.pos_dec_time;

    w.enc.resize(hp.enc_layers);
    for (uint32_t i = 0; i < hp.enc_layers && ok; i++)
        ok = load_layer(tm, "hft.enc." + std::to_string(i), w.enc[i], /*self*/ true, /*cross*/ false);

    w.dec_freq.resize(hp.dec_freq_layers);
    for (uint32_t i = 0; i < hp.dec_freq_layers && ok; i++)
        ok = load_layer(tm, "hft.decfreq." + std::to_string(i), w.dec_freq[i], /*self*/ i > 0, /*cross*/ true);

    w.dec_time.resize(hp.dec_time_layers);
    for (uint32_t i = 0; i < hp.dec_time_layers && ok; i++)
        ok = load_layer(tm, "hft.dectime." + std::to_string(i), w.dec_time[i], /*self*/ true, /*cross*/ false);

    ok = ok && load_linear(tm, "hft.head.onset", w.head_onset);
    ok = ok && load_linear(tm, "hft.head.offset", w.head_offset);
    ok = ok && load_linear(tm, "hft.head.mpe", w.head_mpe);
    ok = ok && load_linear(tm, "hft.head.velocity", w.head_velocity);
    if (!ok) {
        hft_transformer_free(ctx);
        return nullptr;
    }

    w.mel_fb = core_gguf::try_get(tm, "hft.mel_fb");
    w.window = core_gguf::try_get(tm, "hft.window");

    const int n_freqs = (int)hp.n_fft / 2 + 1;
    if (!w.mel_fb) {
        // No fallback worth having: the checkpoint was trained on one
        // particular filterbank (HTK scale, slaney norm, fmin 0, fmax 8000)
        // and getting any one of those wrong costs accuracy silently.
        std::fprintf(stderr, "hft: GGUF has no hft.mel_fb — reconvert with "
                             "models/convert-hft-transformer-to-gguf.py\n");
        hft_transformer_free(ctx);
        return nullptr;
    }
    ctx->mel_fb = hft_to_f32(w.mel_fb);
    if ((int)ctx->mel_fb.size() != (int)hp.n_mels * n_freqs) {
        std::fprintf(stderr, "hft: mel_fb is %zu floats, expected %d\n", ctx->mel_fb.size(), (int)hp.n_mels * n_freqs);
        hft_transformer_free(ctx);
        return nullptr;
    }
    if (w.window) {
        ctx->hann = hft_to_f32(w.window);
    } else {
        ctx->hann.resize(hp.n_fft);
        for (uint32_t i = 0; i < hp.n_fft; i++)
            ctx->hann[i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)i / (float)hp.n_fft);
    }

    if (params.verbosity >= 1) {
        std::fprintf(stderr, "hft: loaded model (%u mels, %u classes, hop %u, %u+%u+%u layers, dim %u)\n", hp.n_mels,
                     hp.classes_num, hp.hop_size, hp.enc_layers, hp.dec_freq_layers, hp.dec_time_layers, hp.hidden);
    }
    return ctx;
}

void hft_transformer_free(struct hft_transformer_ctx* ctx) {
    if (!ctx)
        return;
    if (ctx->alloc_enc)
        ggml_gallocr_free(ctx->alloc_enc);
    if (ctx->alloc_time)
        ggml_gallocr_free(ctx->alloc_time);
    if (ctx->w_buf)
        core_gguf::release_weight_buffer(ctx->w_buf);
    if (ctx->w_buf_cpu)
        core_gguf::release_weight_buffer(ctx->w_buf_cpu);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

uint32_t hft_transformer_sample_rate(const struct hft_transformer_ctx* ctx) {
    return ctx ? ctx->hp.sample_rate : 16000;
}

// ─── Front end ──────────────────────────────────────────────────────────────

// Returns [T, n_mels] row-major, log(mel + 1e-8).
static std::vector<float> hft_log_mel(hft_transformer_ctx* ctx, const float* pcm, int n_samples, int& T_out) {
    hft_bench_stage _b("mel");
    auto& hp = ctx->hp;
    const int n_freqs = (int)hp.n_fft / 2 + 1;

    core_mel::Params p;
    p.n_fft = (int)hp.n_fft;
    p.hop_length = (int)hp.hop_size;
    p.win_length = (int)hp.n_fft;
    p.n_mels = (int)hp.n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.log_guard = core_mel::LogGuard::AddEpsilon; // log(mel + eps), not a clamp
    p.log_eps = hp.mel_eps;
    p.spec_kind = core_mel::SpecKind::Power; // power 2.0
    p.norm = core_mel::Normalization::None;
    p.layout = core_mel::Layout::TimeMels;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Float;
    p.center_pad = true;
    p.center_pad_reflect = false; // librosa pad_mode="constant"

    return core_mel::compute(pcm, n_samples, ctx->hann.data(), (int)hp.n_fft, ctx->mel_fb.data(), n_freqs, hft_fft_r2c,
                             p, T_out);
}

float* hft_transformer_mel(struct hft_transformer_ctx* ctx, const float* pcm, int n_samples, int* out_frames) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;
    int T = 0;
    auto mel = hft_log_mel(ctx, pcm, n_samples, T);
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

// ─── Graph building blocks ──────────────────────────────────────────────────

// Every weight GEMM in this model is position-wise, so the token and batch
// axes can be flattened into one before the multiply and restored after. That
// matters more than it looks: `ggml_mul_mat` loops over ne2×ne3 and issues one
// GEMM call per batch element, so a [256, 256, 32] activation becomes 32
// separate 256×256×256 multiplies that each re-stream the weight matrix,
// where the flattened form is a single 256×8192×256 one that loads it once.
// The reshape is a view — same memory, same arithmetic, same result.
static ggml_tensor* hft_linear_apply(ggml_context* c, const hft_linear& l, ggml_tensor* x) {
    const int64_t n0 = x->ne[0], n1 = x->ne[1], n2 = x->ne[2], n3 = x->ne[3];
    ggml_tensor* flat = (n2 * n3 > 1 && ggml_is_contiguous(x)) ? ggml_reshape_2d(c, x, n0, n1 * n2 * n3) : x;
    ggml_tensor* y = ggml_add(c, ggml_mul_mat(c, l.w, flat), l.b);
    if (flat != x)
        y = ggml_reshape_4d(c, y, y->ne[0], n1, n2, n3);
    return y;
}

static ggml_tensor* hft_layer_norm(ggml_context* c, const hft_layer& l, ggml_tensor* x, float eps) {
    ggml_tensor* h = ggml_norm(c, x, eps);
    h = ggml_mul(c, h, l.ln_w);
    return ggml_add(c, h, l.ln_b);
}

// Multi-head attention. q_in is [H, Nq, B], kv_in is [H, Nk, B]; both the
// batch dimensions must match. Returns [H, Nq, B].
static ggml_tensor* hft_attention(ggml_context* c, const hft_attn& a, ggml_tensor* q_in, ggml_tensor* kv_in, int n_head,
                                  float scale) {
    // Self-attention passes q_in for both, so neither is ever null. The
    // assert is here because the alternative — a layer that claims
    // cross-attention with no memory to attend to — is a null dereference two
    // frames down the stack with nothing to say which layer did it.
    GGML_ASSERT(q_in != nullptr && kv_in != nullptr);
    const int64_t H = q_in->ne[0];
    const int64_t Nq = q_in->ne[1];
    const int64_t B = q_in->ne[2];
    const int64_t Nk = kv_in->ne[1];
    const int64_t hd = H / n_head;

    ggml_tensor* q = hft_linear_apply(c, a.q, q_in);
    ggml_tensor* k = hft_linear_apply(c, a.k, kv_in);
    ggml_tensor* v = hft_linear_apply(c, a.v, kv_in);

    // [hd, n_head, N, B] -> [hd, N, n_head, B]
    q = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, q, hd, n_head, Nq, B), 0, 2, 1, 3));
    k = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, k, hd, n_head, Nk, B), 0, 2, 1, 3));
    // v needs its head-dim axis last for the P·V product: [Nk, hd, n_head, B]
    v = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, v, hd, n_head, Nk, B), 1, 2, 0, 3));

    ggml_tensor* kq = ggml_mul_mat(c, k, q); // [Nk, Nq, n_head, B]
    kq = ggml_soft_max_ext(c, kq, nullptr, scale, 0.0f);
    ggml_tensor* kqv = ggml_mul_mat(c, v, kq);            // [hd, Nq, n_head, B]
    kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3)); // [hd, n_head, Nq, B]
    kqv = ggml_reshape_3d(c, kqv, H, Nq, B);
    return hft_linear_apply(c, a.o, kqv);
}

// x = LN(x + Attn(...)); … ; x = LN(x + FF(x)). `enc` is the cross-attention
// memory. A self-attention-only layer has none, and callers pass null for it.
//
// `enc` is resolved to `x` rather than forwarded raw, so `hft_attention` can
// never receive a null operand. That is not only for cppcheck's benefit,
// though it is the fix for a `ctunullpointer` it reports on the encoder call
// site: the guard that makes the raw form safe today is `l.has_cross`, a
// runtime field set at load time, and anything that ever sets it on a layer
// whose caller has no memory turns the raw form into a null dereference with
// no diagnostic. Resolving here makes that failure a wrong ANSWER rather than
// a crash, and the GGML_ASSERT in hft_attention covers the rest.
static ggml_tensor* hft_apply_layer(ggml_context* c, const hft_layer& l, ggml_tensor* x, ggml_tensor* enc, int n_head,
                                    float scale, float ln_eps) {
    if (l.has_self)
        x = hft_layer_norm(c, l, ggml_add(c, x, hft_attention(c, l.self, x, x, n_head, scale)), ln_eps);
    if (l.has_cross) {
        ggml_tensor* mem = enc ? enc : x;
        x = hft_layer_norm(c, l, ggml_add(c, x, hft_attention(c, l.cross, x, mem, n_head, scale)), ln_eps);
    }
    ggml_tensor* ff = hft_linear_apply(c, l.ff1, x);
    ff = ggml_relu(c, ff);
    ff = hft_linear_apply(c, l.ff2, ff);
    return hft_layer_norm(c, l, ggml_add(c, x, ff), ln_eps);
}

static ggml_context* hft_graph_ctx(hft_transformer_ctx* ctx, size_t bytes) {
    if (ctx->graph_meta.size() < bytes)
        ctx->graph_meta.resize(bytes);
    ggml_init_params ip = {ctx->graph_meta.size(), ctx->graph_meta.data(), true};
    return ggml_init(ip);
}

// ─── Encoder + frequency decoder, one chunk of frames ───────────────────────

// taps: [front_taps, n_mels, n_chunk] — for every (frame, mel bin), the
// front_taps consecutive log-mel values the model's 65-frame window sees.
// Returns [hidden, 88, n_chunk].
static bool hft_encode_chunk(hft_transformer_ctx* ctx, const float* taps, int n_chunk, std::vector<float>& out) {
    auto& hp = ctx->hp;
    auto& w = ctx->weights;
    const int64_t taps_n = hp.front_taps;
    const int64_t bins = hp.n_mels;
    const int64_t H = hp.hidden;
    const int64_t K = hp.classes_num;
    const int n_head = (int)hp.n_heads;
    const float scale = 1.0f / std::sqrt((float)(H / hp.n_heads));
    const float emb_scale = std::sqrt((float)H);

    ggml_context* c = hft_graph_ctx(ctx, 16u * 1024 * 1024);
    if (!c)
        return false;
    ggml_cgraph* gf = ggml_new_graph_custom(c, 8192, false);

    ggml_tensor* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, taps_n, bins, n_chunk);
    ggml_set_name(x, "taps");
    ggml_set_input(x);

    // Fused conv + token embedding, then ×√256 and the frequency positions.
    ggml_tensor* h = hft_linear_apply(c, w.front, x); // [H, bins, n_chunk]
    h = ggml_scale(c, h, emb_scale);
    h = ggml_add(c, h, w.pos_enc_freq);
    for (const auto& l : w.enc)
        h = hft_apply_layer(c, l, h, nullptr, n_head, scale, hp.ln_eps);

    // Frequency decoder: 88 pitch queries, identical for every frame, cross-
    // attending to this frame's 256 encoder tokens. No √256 on the query —
    // the graph adds the cross-attention output straight onto the embedding.
    ggml_tensor* q = ggml_repeat_4d(c, w.pos_dec_freq, H, K, n_chunk, 1);
    for (const auto& l : w.dec_freq)
        q = hft_apply_layer(c, l, q, h, n_head, scale, hp.ln_eps);

    ggml_set_name(q, "out");
    ggml_set_output(q);
    ggml_build_forward_expand(gf, q);

    if (!ctx->alloc_enc)
        ctx->alloc_enc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->alloc_enc || !ggml_gallocr_alloc_graph(ctx->alloc_enc, gf)) {
        std::fprintf(stderr, "hft: encoder graph allocation failed (chunk=%d)\n", n_chunk);
        ggml_free(c);
        return false;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "taps"), taps, 0,
                            (size_t)taps_n * bins * n_chunk * sizeof(float));
    hft_set_threads(ctx);
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "hft: encoder graph compute failed\n");
        ggml_free(c);
        return false;
    }
    ggml_tensor* res = ggml_graph_get_tensor(gf, "out");
    out.resize((size_t)ggml_nelements(res));
    ggml_backend_tensor_get(res, out.data(), 0, out.size() * sizeof(float));
    ggml_free(c);
    return true;
}

// ─── Time decoder + heads, one 128-frame window ─────────────────────────────

struct hft_window_out {
    std::vector<float> onset, offset, mpe; // [n_frame * 88], post-sigmoid
    std::vector<float> velocity;           // [n_frame * 88], argmax bin
};

// pitch_major: [hidden, n_frame, 88] — the frequency decoder's output with
// pitch moved out to the batch axis.
static bool hft_decode_time(hft_transformer_ctx* ctx, const float* pitch_major, hft_window_out& out) {
    auto& hp = ctx->hp;
    auto& w = ctx->weights;
    const int64_t H = hp.hidden;
    const int64_t T = hp.n_frame;
    const int64_t K = hp.classes_num;
    const int64_t VB = hp.velocity_bins;
    const int n_head = (int)hp.n_heads;
    const float scale = 1.0f / std::sqrt((float)(H / hp.n_heads));
    const float emb_scale = std::sqrt((float)H);

    ggml_context* c = hft_graph_ctx(ctx, 16u * 1024 * 1024);
    if (!c)
        return false;
    ggml_cgraph* gf = ggml_new_graph_custom(c, 8192, false);

    ggml_tensor* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, H, T, K);
    ggml_set_name(x, "in");
    ggml_set_input(x);

    ggml_tensor* h = ggml_add(c, ggml_scale(c, x, emb_scale), w.pos_dec_time);
    for (const auto& l : w.dec_time)
        h = hft_apply_layer(c, l, h, nullptr, n_head, scale, hp.ln_eps);

    struct {
        const char* name;
        const hft_linear* lin;
    } heads[] = {
        {"onset", &w.head_onset},
        {"offset", &w.head_offset},
        {"mpe", &w.head_mpe},
        {"velocity", &w.head_velocity},
    };
    for (auto& hd : heads) {
        ggml_tensor* y = hft_linear_apply(c, *hd.lin, h);
        ggml_set_name(y, hd.name);
        ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
    }

    if (!ctx->alloc_time)
        ctx->alloc_time = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->alloc_time || !ggml_gallocr_alloc_graph(ctx->alloc_time, gf)) {
        std::fprintf(stderr, "hft: time-decoder graph allocation failed\n");
        ggml_free(c);
        return false;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "in"), pitch_major, 0, (size_t)H * T * K * sizeof(float));
    hft_set_threads(ctx);
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "hft: time-decoder graph compute failed\n");
        ggml_free(c);
        return false;
    }

    // Each scalar head comes back as [1, T, K]; transpose to [T, K] and
    // apply the sigmoid the reference's `infer.py` omits (its heads are
    // logits and it thresholds them at 0.5 as though they were not).
    std::vector<float> buf;
    auto fetch = [&](const char* name, std::vector<float>& dst, int64_t width) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        buf.resize((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        dst.assign((size_t)T * K, 0.0f);
        if (width == 1) {
            for (int64_t p = 0; p < K; p++)
                for (int64_t tt = 0; tt < T; tt++)
                    dst[(size_t)tt * K + p] = hft_sigmoid(buf[(size_t)p * T + tt]);
        } else {
            for (int64_t p = 0; p < K; p++) {
                for (int64_t tt = 0; tt < T; tt++) {
                    const float* row = buf.data() + ((size_t)p * T + tt) * width;
                    int best = 0;
                    for (int64_t v = 1; v < width; v++)
                        if (row[v] > row[best])
                            best = (int)v;
                    dst[(size_t)tt * K + p] = (float)best;
                }
            }
        }
    };
    fetch("onset", out.onset, 1);
    fetch("offset", out.offset, 1);
    fetch("mpe", out.mpe, 1);
    fetch("velocity", out.velocity, VB);

    ggml_free(c);
    return true;
}

// ─── One forward pass over a whole clip ─────────────────────────────────────

struct hft_heads {
    std::vector<float> onset, offset, mpe, velocity; // [T, 88]
    int T = 0;
};

static bool hft_forward(hft_transformer_ctx* ctx, const float* pcm, int n_samples, hft_heads& out) {
    auto& hp = ctx->hp;
    const int bins = (int)hp.n_mels;
    const int margin = (int)hp.n_margin;
    const int taps = (int)hp.front_taps;
    const int NF = (int)hp.n_frame;
    const int K = (int)hp.classes_num;
    const int H = (int)hp.hidden;

    int T = 0;
    auto mel = hft_log_mel(ctx, pcm, n_samples, T);
    if (T <= 0)
        return false;

    // `infer.py`'s window arithmetic: margin frames of log(mel_eps) at each
    // end, and the tail padded up to a multiple of n_frame so every window is
    // full. The model then answers for the middle n_frame of each 192-frame
    // window, so output frame i of window w is feature frame w*n_frame + i.
    const int total = ((T + NF - 1) / NF) * NF;
    const float minv = std::log(hp.mel_eps);
    std::vector<float> padded((size_t)(total + 2 * margin) * bins, minv);
    std::memcpy(padded.data() + (size_t)margin * bins, mel.data(), (size_t)T * bins * sizeof(float));

    out.T = total;
    out.onset.assign((size_t)total * K, 0.0f);
    out.offset.assign((size_t)total * K, 0.0f);
    out.mpe.assign((size_t)total * K, 0.0f);
    out.velocity.assign((size_t)total * K, 0.0f);

    int chunk = ctx->params.frame_chunk > 0 ? ctx->params.frame_chunk : HFT_FRAME_CHUNK_DEFAULT;
    chunk = std::max(1, std::min(chunk, NF));

    std::vector<float> tapbuf, encbuf, pitch_major((size_t)H * NF * K);
    hft_window_out wout;

    for (int base = 0; base < total; base += NF) {
        hft_bench_stage _b("window");
        for (int c0 = 0; c0 < NF; c0 += chunk) {
            const int nc = std::min(chunk, NF - c0);
            // taps[(f * bins + b) * taps + m] = padded[(base + c0 + f + m) * bins + b]
            tapbuf.resize((size_t)nc * bins * taps);
            for (int f = 0; f < nc; f++) {
                const int g = base + c0 + f;
                for (int b = 0; b < bins; b++) {
                    float* dst = tapbuf.data() + ((size_t)f * bins + b) * taps;
                    const float* src = padded.data() + (size_t)g * bins + b;
                    for (int m = 0; m < taps; m++)
                        dst[m] = src[(size_t)m * bins];
                }
            }
            if (!hft_encode_chunk(ctx, tapbuf.data(), nc, encbuf))
                return false;
            // encbuf is [H, 88, nc]; the time decoder wants [H, n_frame, 88].
            for (int f = 0; f < nc; f++)
                for (int p = 0; p < K; p++)
                    std::memcpy(pitch_major.data() + ((size_t)p * NF + c0 + f) * H,
                                encbuf.data() + ((size_t)f * K + p) * H, (size_t)H * sizeof(float));
        }
        if (!hft_decode_time(ctx, pitch_major.data(), wout))
            return false;
        const int n = std::min(NF, total - base);
        const size_t off = (size_t)base * K;
        std::memcpy(out.onset.data() + off, wout.onset.data(), (size_t)n * K * sizeof(float));
        std::memcpy(out.offset.data() + off, wout.offset.data(), (size_t)n * K * sizeof(float));
        std::memcpy(out.mpe.data() + off, wout.mpe.data(), (size_t)n * K * sizeof(float));
        std::memcpy(out.velocity.data() + off, wout.velocity.data(), (size_t)n * K * sizeof(float));
    }
    return true;
}

// ─── Note decoding (preprocess/midi.py) ─────────────────────────────────────

struct hft_event {
    int frame;
    float time;
};

// `detect_event`. A frame is an event when it is at or above the threshold and
// is a local maximum in the WEAK sense — scanning outward in each direction,
// the first strictly different neighbour is smaller. The time is then refined
// between the neighbours, which is where hFT gets onset resolution finer than
// its 16 ms frame.
static void hft_detect_event(const std::vector<float>& data, int T, int K, int pitch, float threshold, float hop_sec,
                             std::vector<hft_event>& out) {
    out.clear();
    auto at = [&](int i) { return data[(size_t)i * K + pitch]; };
    for (int i = 0; i < T; i++) {
        const float v = at(i);
        if (v < threshold)
            continue;
        bool ok = true;
        for (int ii = i - 1; ii >= 0; ii--) {
            if (v > at(ii))
                break;
            if (v < at(ii)) {
                ok = false;
                break;
            }
        }
        if (!ok)
            continue;
        for (int ii = i + 1; ii < T; ii++) {
            if (v > at(ii))
                break;
            if (v < at(ii)) {
                ok = false;
                break;
            }
        }
        if (!ok)
            continue;
        float t = (float)i * hop_sec;
        if (i > 0 && i < T - 1) {
            const float l = at(i - 1), r = at(i + 1);
            if (l > r)
                t = (float)i * hop_sec - hop_sec * 0.5f * (l - r) / (v - r);
            else if (r > l)
                t = (float)i * hop_sec + hop_sec * 0.5f * (r - l) / (v - l);
        }
        out.push_back({i, t});
    }
}

// `convert_label_to_note` with `mode_offset='shorter'` and
// `mode_velocity='ignore_zero'`.
static void hft_extract_notes(const hft_heads& h, const hft_hparams& hp, const hft_transformer_params& p,
                              std::vector<hft_transformer_note_event>& notes) {
    const int K = (int)hp.classes_num;
    const int T = h.T;
    const float hop_sec = (float)hp.hop_size / (float)hp.sample_rate;

    std::vector<hft_event> onsets, offsets;
    for (int pitch = 0; pitch < K; pitch++) {
        hft_detect_event(h.onset, T, K, pitch, p.onset_threshold, hop_sec, onsets);
        if (onsets.empty())
            continue;
        hft_detect_event(h.offset, T, K, pitch, p.offset_threshold, hop_sec, offsets);

        for (size_t k = 0; k < onsets.size(); k++) {
            const int loc_onset = onsets[k].frame;
            const float time_onset = onsets[k].time;
            int loc_next;
            float time_next;
            if (k + 1 < onsets.size()) {
                loc_next = onsets[k + 1].frame;
                time_next = onsets[k + 1].time;
            } else {
                loc_next = T;
                time_next = (float)(loc_next - 1) * hop_sec;
            }

            int loc_offset = loc_onset + 1;
            float time_offset = 0.0f;
            bool flag_offset = false;
            for (const auto& e : offsets) {
                if (loc_onset < e.frame) {
                    loc_offset = e.frame;
                    time_offset = e.time;
                    flag_offset = true;
                    break;
                }
            }
            if (loc_offset > loc_next) {
                loc_offset = loc_next;
                time_offset = time_next;
            }

            int loc_mpe = loc_onset + 1;
            float time_mpe = 0.0f;
            bool flag_mpe = false;
            for (int ii = loc_onset + 1; ii < std::min(loc_next, T); ii++) {
                if (h.mpe[(size_t)ii * K + pitch] < p.mpe_threshold) {
                    loc_mpe = ii;
                    flag_mpe = true;
                    time_mpe = (float)loc_mpe * hop_sec;
                    break;
                }
            }

            float offset_value;
            if (!flag_offset && !flag_mpe)
                offset_value = time_next;
            else if (flag_offset && !flag_mpe)
                offset_value = time_offset;
            else if (!flag_offset && flag_mpe)
                offset_value = time_mpe;
            else
                offset_value = (loc_offset <= loc_mpe) ? time_offset : time_mpe;

            const int vel = (int)std::lround(h.velocity[(size_t)loc_onset * K + pitch]);
            // `mode_velocity='ignore_zero'`: drop the note outright. See the
            // header — this is hFT's real precision filter, measured at 52.2%
            // F1 against 52.1% for the best thresholded arm without it.
            if (p.ignore_zero_velocity && vel <= 0)
                continue;

            // "a re-onset of the same pitch ends the previous note": the
            // original truncates the note two back when the new one starts
            // before it ends.
            if (!notes.empty() && notes.back().midi_note == (int)hp.begin_note + pitch &&
                time_onset < notes.back().offset_time) {
                notes.back().offset_time = time_onset;
            }
            notes.push_back({time_onset, offset_value, (int)hp.begin_note + pitch, std::max(0, std::min(127, vel))});
        }
    }
    std::stable_sort(notes.begin(), notes.end(),
                     [](const hft_transformer_note_event& a, const hft_transformer_note_event& b) {
                         return a.onset_time < b.onset_time;
                     });
}

// ─── Public entry point ─────────────────────────────────────────────────────

int hft_transformer_transcribe(struct hft_transformer_ctx* ctx, const float* pcm, int n_samples,
                               struct hft_transformer_result* result) {
    if (!ctx || !pcm || n_samples <= 0 || !result)
        return -1;
    std::memset(result, 0, sizeof(*result));

    hft_heads all;
    if (!hft_forward(ctx, pcm, n_samples, all))
        return -2;

    std::vector<hft_transformer_note_event> notes;
    hft_extract_notes(all, ctx->hp, ctx->params, notes);

    result->n_notes = (int)notes.size();
    result->n_frames = all.T;
    result->n_classes = (int)ctx->hp.classes_num;
    if (!notes.empty()) {
        result->note_events =
            (hft_transformer_note_event*)std::malloc(notes.size() * sizeof(hft_transformer_note_event));
        if (!result->note_events)
            return -3;
        std::memcpy(result->note_events, notes.data(), notes.size() * sizeof(hft_transformer_note_event));
    }

    if (ctx->params.verbosity >= 2) {
        auto dup = [&](const std::vector<float>& v) -> float* {
            float* b = (float*)std::malloc(v.size() * sizeof(float));
            if (b)
                std::memcpy(b, v.data(), v.size() * sizeof(float));
            return b;
        };
        result->onset_output = dup(all.onset);
        result->offset_output = dup(all.offset);
        result->mpe_output = dup(all.mpe);
        result->velocity_output = dup(all.velocity);
    }
    return 0;
}

void hft_transformer_result_free(struct hft_transformer_result* result) {
    if (!result)
        return;
    std::free(result->note_events);
    std::free(result->onset_output);
    std::free(result->offset_output);
    std::free(result->mpe_output);
    std::free(result->velocity_output);
    std::memset(result, 0, sizeof(*result));
}
