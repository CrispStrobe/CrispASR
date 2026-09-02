// src/mel_band_roformer.cpp — Mel-Band RoFormer source separation (§248).
//
// Phase 1 (this file): GGUF loader (hparams + baked band-layout aux arrays +
// weight map), the STFT front-end that matches torch.stft(center=True) and the
// binary-band gather, and the front-end half of the diff harness
// (input_audio -> stft_packed -> band_gathered). The transformer graph, mask
// estimator, scatter-average, complex mask and iSTFT are Phase 2 — built and
// validated stage-by-stage against the reference fixture (ref_mbr.gguf), first
// divergence = the bug.
//
// Blueprint: MIT lucidrains/BS-RoFormer. Weights: KimberleyJSN/melbandroformer
// (MIT). Reference pinned at bs-roformer==0.3.10. See docs/mel-band-roformer/PLAN.md.

#include "mel_band_roformer.h"

#include "ggml-backend.h"
#include "ggml-alloc.h" // ggml_gallocr_* (Change 176: band-split ggml graph path)
#include "ggml-cpu.h"
#include "ggml.h"

#include "core/fft.h"         // fft_radix2_wrapper (r2c, interleaved full spectrum)
#include "core/ggml_cpu_backend.h" // core_cpu_backend::init/is_cpu (Change 176)
#include "core/gguf_loader.h" // core_gguf::{open_metadata,kv_u32,load_weights}
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (Change 176, htdemucs pattern)
#include "core/istft.h"       // core_istft::istft (torch center=True match)
#include "core/wav_reader.h"  // crispasr::core::read_wav_mono_pcm16 (Change 176 parity)

// BLAS for the linear() SGEMM. #296: the forward is ~264 GFLOP of matmul; without
// a BLAS backend linear() falls back to a scalar loop that took ~24 min on an 11s
// clip (Linux/Windows), while macOS was fast via Accelerate. Use the portable
// cblas the same way cohere/crispasr-core do: Accelerate on Apple, <cblas.h>
// (OpenBLAS/MKL) elsewhere when the build found one (HAVE_BLAS).
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h> // cblas + vDSP, no external deps
#elif defined(HAVE_BLAS)
#include <cblas.h>
#endif

// #296: run_time/run_freq parallelise the band/time loops with OpenMP and each
// block calls BLAS — a threaded BLAS would nest and oversubscribe cores. Pin BLAS
// to one thread. Gated on CRISPASR_MBR_OPENBLAS (set by CMake ONLY when OpenBLAS
// is the linked BLAS), so the symbol is guaranteed present — no __attribute__(
// (weak)), which MSVC rejects. Reference cblas / MKL / Accelerate skip this
// (single-threaded or self-managed).
#if defined(CRISPASR_MBR_OPENBLAS)
extern "C" void openblas_set_num_threads(int);
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ---------------------------------------------------------------------------
// Model context
// ---------------------------------------------------------------------------

struct mel_band_roformer_hparams {
    int dim = 384;
    int depth = 6;
    int heads = 8;
    int dim_head = 64;
    int num_bands = 60;
    int num_stems = 1;
    int time_transformer_depth = 1;
    int freq_transformer_depth = 1;
    int mask_estimator_depth = 2;
    int stereo = 1;
    int audio_channels = 2;
    int sample_rate = 44100;
    int n_fft = 2048;
    int hop = 441;
    int win = 2048;
    int normalized = 0;
};

struct mel_band_roformer_context {
    mel_band_roformer_hparams hp;
    mel_band_roformer_params params{};

    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;

    // Change 176: band-split runs as a ggml graph on `backend` (GPU-capable)
    // when use_graph is set; otherwise the validated CPU reference path runs.
    bool use_graph = false;
    bool is_gpu = false;

    // Baked band layout (from the converter's aux.* int32 tensors).
    std::vector<int32_t> freq_indices;       // length = sum over bands of 2*nfreq*ch? no: N gather idx
    std::vector<int32_t> num_bands_per_freq; // length = n_freqs (overlap denominator)
    std::vector<int32_t> num_freqs_per_band; // length = num_bands
    // Per-band packed-input width = 2 (complex) * num_freqs_per_band[b] * channels.
    std::vector<int> band_width;

    std::vector<std::string> source_names_storage;
    std::vector<const char*> source_names_c;

    int n_freqs() const { return hp.n_fft / 2 + 1; }
};

// ---------------------------------------------------------------------------
// STFT front-end (CPU) — matches torch.stft(center=True, Hann-periodic)
// ---------------------------------------------------------------------------

namespace {

// Hann periodic window of length N (torch.hann_window default: periodic=True):
//   w[n] = 0.5 - 0.5*cos(2*pi*n / N)
void hann_periodic(int N, std::vector<float>& w) {
    w.resize(N);
    for (int n = 0; n < N; n++)
        w[n] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)n / (float)N);
}

// torch center=True reflect-pad by n_fft/2 on both ends (reflect excludes the
// edge sample, matching numpy/torch 'reflect').
void reflect_pad(const float* x, int n, int pad, std::vector<float>& out) {
    out.resize((size_t)n + 2 * pad);
    for (int i = 0; i < pad; i++)
        out[i] = x[pad - i]; // reflect: x[pad], x[pad-1], ... excludes x[0] mirror center
    for (int i = 0; i < n; i++)
        out[pad + i] = x[i];
    for (int i = 0; i < pad; i++)
        out[pad + n + i] = x[n - 2 - i];
}

// Frame count for center=True: 1 + n_samples/hop (padded length n + n_fft,
// frames = 1 + (padded - n_fft)/hop = 1 + n/hop).
int stft_n_frames(int n_samples, int hop) {
    return 1 + n_samples / hop;
}

// One channel -> complex STFT. Output `spec` is [n_freqs][T][2] laid out as
// spec[(f*T + t)*2 + {0,1}]. n_freqs = n_fft/2+1.
void stft_one_channel(const float* x, int n, int n_fft, int hop, const std::vector<float>& window, int T, int n_freqs,
                      std::vector<float>& spec) {
    const int pad = n_fft / 2;
    std::vector<float> xp;
    reflect_pad(x, n, pad, xp);
    const int padded = (int)xp.size();

    spec.assign((size_t)n_freqs * T * 2, 0.0f);
    std::vector<float> frame(n_fft), full(2 * n_fft);
    for (int t = 0; t < T; t++) {
        const int start = t * hop;
        for (int i = 0; i < n_fft; i++) {
            const int idx = start + i;
            frame[i] = (idx < padded ? xp[idx] : 0.0f) * window[i];
        }
        core_fft::fft_radix2_wrapper(frame.data(), n_fft, full.data()); // full[2k]=re, full[2k+1]=im
        for (int f = 0; f < n_freqs; f++) {
            spec[((size_t)f * T + t) * 2 + 0] = full[2 * f + 0];
            spec[((size_t)f * T + t) * 2 + 1] = full[2 * f + 1];
        }
    }
}

// Build the packed STFT `(f*s, t, 2)` frequency-major / channel-fastest:
//   packed[((f*ch + s)*T + t)*2 + c]. `chan_spec[s]` is one channel's
// [n_freqs][T][2] buffer.
void pack_stft(const std::vector<std::vector<float>>& chan_spec, int n_freqs, int T, int channels,
               std::vector<float>& packed) {
    packed.assign((size_t)n_freqs * channels * T * 2, 0.0f);
    for (int f = 0; f < n_freqs; f++)
        for (int s = 0; s < channels; s++) {
            const int row = f * channels + s;
            for (int t = 0; t < T; t++) {
                packed[((size_t)row * T + t) * 2 + 0] = chan_spec[s][((size_t)f * T + t) * 2 + 0];
                packed[((size_t)row * T + t) * 2 + 1] = chan_spec[s][((size_t)f * T + t) * 2 + 1];
            }
        }
}

// Gather the packed rows named by freq_indices and fold complex into the
// feature axis: band_gathered[t][k*2 + c] where k indexes freq_indices.
// Output shape (T, N*2) with N = freq_indices.size().
void band_gather(const std::vector<float>& packed, const std::vector<int32_t>& freq_indices, int T,
                 std::vector<float>& out) {
    const int N = (int)freq_indices.size();
    out.assign((size_t)T * N * 2, 0.0f);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < N; k++) {
            const int row = freq_indices[k];
            out[((size_t)t * N + k) * 2 + 0] = packed[((size_t)row * T + t) * 2 + 0];
            out[((size_t)t * N + k) * 2 + 1] = packed[((size_t)row * T + t) * 2 + 1];
        }
}

// Read an int32 aux tensor from the loaded weights into a host vector.
bool read_i32(core_gguf::WeightLoad& wl, const char* name, std::vector<int32_t>& out) {
    auto it = wl.tensors.find(name);
    if (it == wl.tensors.end() || !it->second)
        return false;
    ggml_tensor* t = it->second;
    const int64_t n = ggml_nelements(t);
    out.resize((size_t)n);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(int32_t));
    return true;
}

// Read a weight tensor (F32 or F16) into a host f32 vector. Returns false if
// missing or an unhandled dtype.
bool read_f32(core_gguf::WeightLoad& wl, const std::string& name, std::vector<float>& out) {
    auto it = wl.tensors.find(name);
    if (it == wl.tensors.end() || !it->second)
        return false;
    ggml_tensor* t = it->second;
    const int64_t n = ggml_nelements(t);
    out.resize((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t)n);
        ggml_backend_tensor_get(t, tmp.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = ggml_fp16_to_fp32(tmp[(size_t)i]);
    } else {
        return false;
    }
    return true;
}

// lucidrains RMSNorm: F.normalize(x, dim=-1) * sqrt(dim) * gamma. Algebraically
// x / sqrt(mean(x^2)) * gamma, but the eps lives INSIDE the L2 norm (F.normalize
// default 1e-12), not added to the mean-square — match that exactly.
void rms_norm_inplace(float* x, int dim, const float* gamma) {
    double ss = 0;
    for (int i = 0; i < dim; i++)
        ss += (double)x[i] * x[i];
    const double denom = std::sqrt(ss) + 1e-12; // F.normalize eps
    const double scale = std::sqrt((double)dim);
    for (int i = 0; i < dim; i++)
        x[i] = (float)((double)x[i] / denom * scale) * gamma[i];
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

mel_band_roformer_params mel_band_roformer_default_params(void) {
    mel_band_roformer_params p;
    p.n_threads = 0;
    p.use_gpu = false; // CPU front-end in Phase 1
    p.gpu_device = 0;
    return p;
}

mel_band_roformer_context* mel_band_roformer_init_from_file(const char* model_path, mel_band_roformer_params params) {
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "mel_band_roformer: cannot open GGUF '%s'\n", model_path);
        return nullptr;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (arch != "mel-band-roformer") {
        fprintf(stderr, "mel_band_roformer: GGUF arch is '%s', expected 'mel-band-roformer'\n", arch.c_str());
        core_gguf::free_metadata(meta);
        return nullptr;
    }

    auto* ctx = new mel_band_roformer_context();
    ctx->params = params;
    auto& hp = ctx->hp;
    hp.dim = (int)core_gguf::kv_u32(meta, "mel-band-roformer.dim", hp.dim);
    hp.depth = (int)core_gguf::kv_u32(meta, "mel-band-roformer.depth", hp.depth);
    hp.heads = (int)core_gguf::kv_u32(meta, "mel-band-roformer.heads", hp.heads);
    hp.dim_head = (int)core_gguf::kv_u32(meta, "mel-band-roformer.dim_head", hp.dim_head);
    hp.num_bands = (int)core_gguf::kv_u32(meta, "mel-band-roformer.num_bands", hp.num_bands);
    hp.num_stems = (int)core_gguf::kv_u32(meta, "mel-band-roformer.num_stems", hp.num_stems);
    hp.time_transformer_depth =
        (int)core_gguf::kv_u32(meta, "mel-band-roformer.time_transformer_depth", hp.time_transformer_depth);
    hp.freq_transformer_depth =
        (int)core_gguf::kv_u32(meta, "mel-band-roformer.freq_transformer_depth", hp.freq_transformer_depth);
    hp.mask_estimator_depth =
        (int)core_gguf::kv_u32(meta, "mel-band-roformer.mask_estimator_depth", hp.mask_estimator_depth);
    hp.stereo = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stereo", hp.stereo);
    hp.audio_channels = (int)core_gguf::kv_u32(meta, "mel-band-roformer.audio_channels", hp.audio_channels);
    hp.sample_rate = (int)core_gguf::kv_u32(meta, "mel-band-roformer.sample_rate", hp.sample_rate);
    hp.n_fft = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stft_n_fft", hp.n_fft);
    hp.hop = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stft_hop_length", hp.hop);
    hp.win = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stft_win_length", hp.win);
    hp.normalized = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stft_normalized", hp.normalized);
    core_gguf::free_metadata(meta);

    // Change 176: backend selection honours params.use_gpu + CRISPASR_MELBAND_GPU,
    // mirroring the htdemucs pattern (src/htdemucs.cpp:635-662). The ggml graph
    // path is the gate to the GPU backend: want_gpu WITHOUT the graph path would
    // put the weights on the GPU and pay a device->host read on every scalar
    // kernel (htdemucs lesson). CRISPASR_MELBAND_GGML=1 enables the graph band-
    // split; CRISPASR_MELBAND_GPU=1 forces the GPU backend (falls back to CPU on
    // GPU-less hosts), =0 forces CPU. Default: CPU backend + CPU reference path
    // (no behavior change for existing users until parity is green, E2).
    const char* ggml_env = getenv("CRISPASR_MELBAND_GGML");
    const bool want_graph = ggml_env && ggml_env[0] && atoi(ggml_env) != 0;
    bool want_gpu = false;
    const char* gpu_env = getenv("CRISPASR_MELBAND_GPU");
    if (gpu_env && gpu_env[0])
        want_gpu = atoi(gpu_env) != 0;
    if (params.use_gpu)
        want_gpu = true;
    if (want_graph && want_gpu) {
        ctx->backend = crispasr_init_gpu_backend();
        if (!ctx->backend)
            ctx->backend = core_cpu_backend::init();
    } else {
        ctx->backend = core_cpu_backend::init();
    }
    if (!ctx->backend) {
        fprintf(stderr, "mel_band_roformer: backend init failed\n");
        delete ctx;
        return nullptr;
    }
    ctx->use_graph = want_graph;
    ctx->is_gpu = !core_cpu_backend::is_cpu(ctx->backend);
    fprintf(stderr, "mel_band_roformer: backend = %s (graph=%d, gpu=%d)\n", ggml_backend_name(ctx->backend),
            ctx->use_graph ? 1 : 0, ctx->is_gpu ? 1 : 0);
    if (!core_gguf::load_weights(model_path, ctx->backend, "mel_band_roformer", ctx->weights)) {
        fprintf(stderr, "mel_band_roformer: failed to load weights from '%s'\n", model_path);
        mel_band_roformer_free(ctx);
        return nullptr;
    }

    if (!read_i32(ctx->weights, "aux.freq_indices", ctx->freq_indices) ||
        !read_i32(ctx->weights, "aux.num_bands_per_freq", ctx->num_bands_per_freq) ||
        !read_i32(ctx->weights, "aux.num_freqs_per_band", ctx->num_freqs_per_band)) {
        fprintf(stderr, "mel_band_roformer: GGUF missing baked aux.* band-layout arrays\n");
        mel_band_roformer_free(ctx);
        return nullptr;
    }
    ctx->band_width.resize(ctx->num_freqs_per_band.size());
    for (size_t b = 0; b < ctx->num_freqs_per_band.size(); b++)
        ctx->band_width[b] = 2 * ctx->num_freqs_per_band[b] * hp.audio_channels;

    // Stem names: vocals model emits {vocals, other}. Generic fallback stemN.
    if (hp.num_stems == 1) {
        ctx->source_names_storage = {"vocals", "other"};
    } else {
        for (int i = 0; i < hp.num_stems; i++)
            ctx->source_names_storage.push_back("stem" + std::to_string(i));
    }
    for (auto& s : ctx->source_names_storage)
        ctx->source_names_c.push_back(s.c_str());

    return ctx;
}

void mel_band_roformer_free(mel_band_roformer_context* ctx) {
    if (!ctx)
        return;
    if (ctx->weights.buf)
        core_gguf::release_weight_buffer(ctx->weights.buf);
    if (ctx->weights.ctx)
        ggml_free(ctx->weights.ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

int mel_band_roformer_sample_rate(const mel_band_roformer_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}
int mel_band_roformer_n_sources(const mel_band_roformer_context* ctx) {
    return ctx ? (int)ctx->source_names_storage.size() : 0;
}
const char* mel_band_roformer_source_name(const mel_band_roformer_context* ctx, int idx) {
    if (!ctx || idx < 0 || idx >= (int)ctx->source_names_storage.size())
        return nullptr;
    return ctx->source_names_storage[idx].c_str();
}

void mel_band_roformer_result_free(mel_band_roformer_result* r) {
    if (!r)
        return;
    if (r->sources) {
        for (int s = 0; s < r->n_sources; s++)
            free(r->sources[s]);
        free(r->sources);
    }
    free(r->source_names);
    free(r);
}

// ---------------------------------------------------------------------------
// Diff harness — Phase 1: front-end stages (input_audio, stft_packed,
// band_gathered) vs the reference fixture. Later stages are reported PENDING.
// ---------------------------------------------------------------------------

namespace {

double cosine(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0 || nb == 0)
        return (na == 0 && nb == 0) ? 1.0 : 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

double max_abs_diff(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

double l2_norm(const float* a, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++)
        s += (double)a[i] * a[i];
    return std::sqrt(s);
}

// Fetch a reference tensor's data as a flat float vector.
bool ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out, int64_t& nelem) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end() || !it->second)
        return false;
    ggml_tensor* t = it->second;
    nelem = ggml_nelements(t);
    out.resize((size_t)nelem);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t)nelem * sizeof(float));
    return true;
}

// BandSplit: split the (T, sum(band_width)) gathered tensor into per-band
// chunks, RMSNorm each, project to `dim` via the band's Linear, stack ->
// (T, num_bands, dim). Returns false if any band weight is missing. Contiguous
// per-band because freq_indices is band-major (see the PLAN band-layout note).
bool band_split_cpu(core_gguf::WeightLoad& mw, const std::vector<float>& gathered, const std::vector<int>& band_width,
                    int T, int dim, std::vector<float>& out) {
    const int nb = (int)band_width.size();
    out.assign((size_t)T * nb * dim, 0.0f);
    std::vector<float> gamma, wt, bias, x;
    for (int b = 0; b < nb; b++) {
        const int din = band_width[b];
        const std::string pre = "band_split.to_features." + std::to_string(b);
        if (!read_f32(mw, pre + ".0.gamma", gamma) || !read_f32(mw, pre + ".1.weight", wt) ||
            !read_f32(mw, pre + ".1.bias", bias))
            return false;
        // column offset of band b within the gathered feature axis.
        int off = 0;
        for (int j = 0; j < b; j++)
            off += band_width[j];
        x.resize(din);
        for (int t = 0; t < T; t++) {
            const float* g = gathered.data() + ((size_t)t * (gathered.size() / T)) + off;
            for (int i = 0; i < din; i++)
                x[i] = g[i];
            rms_norm_inplace(x.data(), din, gamma.data());
            // Linear: out[o] = sum_i wt[o*din + i] * x[i] + bias[o]
            float* o = out.data() + ((size_t)t * nb + b) * dim;
            for (int oi = 0; oi < dim; oi++) {
                double acc = bias[oi];
                const float* wrow = wt.data() + (size_t)oi * din;
                for (int i = 0; i < din; i++)
                    acc += (double)wrow[i] * x[i];
                o[oi] = (float)acc;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Change 176 (Phase 1): band-split encoder as a ggml graph, cleanroom port
// following the htdemucs graph pattern (src/htdemucs.cpp). Runs on
// ctx->backend (CPU or GPU); produces the SAME output layout as band_split_cpu
// ((T, nb, dim) row-major — see the flat-layout note below), so the parity test
// compares the two functions element-for-element on identical input.
//
// Flat-layout equivalence: the graph output is a (dim, nb, T) ggml tensor
// (ne[0]=dim fastest). Its flat index is o + dim*(b + nb*t) — byte-identical to
// band_split_cpu's row-major (T, nb, dim) index ((t*nb + b)*dim + o). The graph
// result can therefore be handed straight to the transformer stack unchanged.
// ---------------------------------------------------------------------------
bool band_split_graph(mel_band_roformer_context* ctx, const std::vector<float>& gathered, int T,
                      std::vector<float>& out) {
    const auto& hp = ctx->hp;
    const int nb = (int)ctx->band_width.size();
    const int dim = hp.dim;
    const int N2 = (int)ctx->freq_indices.size() * 2; // gathered feature axis width

    if ((int64_t)gathered.size() != (int64_t)T * N2)
        return false;

    // Per band: view + rms_norm (5 ops) + linear (mul_mat + add) + reshape ≈ 9
    // tensors; plus ~nb concat nodes, the input and the output.
    const size_t n_nodes = 16 * (size_t)nb + 64;
    ggml_init_params gparams = {
        /*.mem_size   = */ ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom((size_t)n_nodes, false),
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context* gctx = ggml_init(gparams);
    if (!gctx)
        return false;

    // Input: gathered is (T, 2N) row-major; as a ggml (2N, T) tensor the flat
    // memory order (ne[0] fastest) is identical, so no copy/transpose is needed.
    ggml_tensor* in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, N2, T);
    ggml_set_name(in, "mbr_band_split_in");
    ggml_set_input(in);

    std::vector<ggml_tensor*> bands;
    bands.reserve((size_t)nb);
    std::vector<ggml_tensor*> sqrt_dim_ts;
    sqrt_dim_ts.reserve((size_t)nb);

    // RMSNorm constants (F.normalize eps, sqrt(dim) scale) as 1-element INPUT
    // tensors: ggml_new_f32 asserts !no_alloc, and the graph context must be
    // no_alloc so gallocr can place intermediates on the backend buffer. Values
    // are written once via ggml_backend_tensor_set before compute (htdemucs
    // pattern). Broadcast to (1, T) via ggml_can_repeat in the binops below.
    ggml_tensor* eps_t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
    ggml_set_name(eps_t, "mbr_rms_eps");
    ggml_set_input(eps_t);

    int off = 0;
    for (int b = 0; b < nb; b++) {
        const int din = ctx->band_width[b];
        const std::string pre = "band_split.to_features." + std::to_string(b);

        auto it_g = ctx->weights.tensors.find(pre + ".0.gamma");
        auto it_w = ctx->weights.tensors.find(pre + ".1.weight");
        auto it_b = ctx->weights.tensors.find(pre + ".1.bias");
        if (it_g == ctx->weights.tensors.end() || it_w == ctx->weights.tensors.end() ||
            it_b == ctx->weights.tensors.end() || !it_g->second || !it_w->second || !it_b->second) {
            ggml_free(gctx);
            return false;
        }

        // Band b's slice: rows [off, off+din) of the (2N, T) input.
        ggml_tensor* xv = ggml_view_2d(gctx, in, din, T, in->nb[1], (size_t)off * sizeof(float));
        off += din;

        // RMSNorm — exact CPU formula: y = x * sqrt(din) / (sqrt(sum(x^2)) + eps) * gamma.
        // NOTE: the sqrt scale uses the BAND's input width din (CPU rms_norm_inplace
        // gets din), NOT the model dim — a global sqrt(hp.dim) scaled every band
        // differently and broke parity (cos=0.877, |graph|/|cpu| ~2.1).
        ggml_tensor* sq = ggml_mul(gctx, xv, xv);                     // (din, T)
        ggml_tensor* ss = ggml_sum_rows(gctx, sq);                    // (1, T)
        ggml_tensor* rt = ggml_sqrt(gctx, ss);                        // (1, T)
        ggml_tensor* denom = ggml_add(gctx, rt, eps_t);               // (1, T)
        // div(a,b) takes a's shape and repeats b to it — repeat sqrt_dim to (1,T) first.
        ggml_tensor* sqrt_dim_t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
        ggml_set_name(sqrt_dim_t, ("mbr_rms_sqrt_dim_" + std::to_string(b)).c_str());
        ggml_set_input(sqrt_dim_t);
        sqrt_dim_ts.push_back(sqrt_dim_t);
        ggml_tensor* scale = ggml_div(gctx, ggml_repeat(gctx, sqrt_dim_t, denom), denom); // (1, T)
        ggml_tensor* xn = ggml_mul(gctx, xv, scale);                  // (din, T) broadcast
        ggml_tensor* y = ggml_mul(gctx, xn, it_g->second);            // * gamma (din, 1) broadcast

        // Linear: mul_mat(weight (din, dim), y (din, T)) -> (dim, T); + bias
        ggml_tensor* proj = ggml_mul_mat(gctx, it_w->second, y);
        ggml_tensor* bout = ggml_add(gctx, proj, it_b->second);

        // (dim, 1, T) for the band-axis concat below.
        ggml_tensor* b3 = ggml_reshape_3d(gctx, bout, dim, 1, T);
        ggml_set_name(b3, ("band_split." + std::to_string(b)).c_str());
        bands.push_back(b3);
    }

    // Balanced concat along the band axis (BSRoformer ConcatBalanced pattern) ->
    // (dim, nb, T). Its flat order equals (T, nb, dim) row-major (see above).
    auto concat_balanced = [&](auto&& self, size_t lo, size_t hi) -> ggml_tensor* {
        if (hi - lo == 1)
            return bands[lo];
        const size_t mid = lo + (hi - lo) / 2;
        ggml_tensor* a = self(self, lo, mid);
        ggml_tensor* b = self(self, mid, hi);
        return ggml_concat(gctx, a, b, 1);
    };
    ggml_tensor* out3 = concat_balanced(concat_balanced, 0, bands.size());
    ggml_set_name(out3, "mbr_band_split_out");
    ggml_set_output(out3);

    ggml_cgraph* gf = ggml_new_graph_custom(gctx, (size_t)n_nodes, false);
    ggml_build_forward_expand(gf, out3);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "mel_band_roformer: band_split_graph gallocr alloc failed\n");
        if (alloc)
            ggml_gallocr_free(alloc);
        ggml_free(gctx);
        return false;
    }

    ggml_backend_tensor_set(in, gathered.data(), 0, gathered.size() * sizeof(float));
    const float eps = 1e-12f; // F.normalize eps — see rms_norm_inplace
    ggml_backend_tensor_set(eps_t, &eps, 0, sizeof(float));
    for (int b = 0; b < nb; b++) {
        // Per-band sqrt(din): the CPU reference scales RMSNorm by sqrt(din_b),
        // not sqrt(hp.dim) — see the formula note above.
        const float sqrt_dim = std::sqrt((float)ctx->band_width[b]);
        ggml_backend_tensor_set(sqrt_dim_ts[b], &sqrt_dim, 0, sizeof(float));
    }
    ggml_backend_graph_compute(ctx->backend, gf);

    const int64_t n = ggml_nelements(out3);
    out.resize((size_t)n);
    ggml_backend_tensor_get(out3, out.data(), 0, (size_t)n * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    return true;
}
void linear(const std::vector<float>& x, int T, int din, const std::vector<float>& W, const std::vector<float>* bias,
            int dout, std::vector<float>& y) {
    y.assign((size_t)T * dout, 0.0f);
#if defined(HAVE_BLAS)
    // y = x @ W^T (x is T x din row-major, W is dout x din row-major). This SGEMM
    // is ~264 GFLOP for a full clip and is the entire reason --separate was fast
    // on macOS (Accelerate) but "hung" for ~24 min elsewhere (#296) — route it
    // through cblas on every platform that has a BLAS.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, dout, din, 1.0f, x.data(), din, W.data(), din, 0.0f,
                y.data(), dout);
    if (bias)
        for (int t = 0; t < T; t++)
            for (int o = 0; o < dout; o++)
                y[(size_t)t * dout + o] += (*bias)[o];
#else
    // No-BLAS fallback: float (not double) accumulation — the torch reference is
    // float32, so this matches it and vectorizes; still far slower than a BLAS.
    for (int t = 0; t < T; t++) {
        const float* xr = x.data() + (size_t)t * din;
        float* yr = y.data() + (size_t)t * dout;
        for (int o = 0; o < dout; o++) {
            float acc = bias ? (*bias)[o] : 0.0f;
            const float* wr = W.data() + (size_t)o * din;
            for (int i = 0; i < din; i++)
                acc += wr[i] * xr[i];
            yr[o] = acc;
        }
    }
#endif
}

// Exact GELU (nn.GELU default, erf form): 0.5*x*(1+erf(x/sqrt2)).
void gelu_erf_inplace(std::vector<float>& v) {
    for (auto& x : v)
        x = 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

// RMSNorm each of T rows of width dim, in place.
void rms_rows(std::vector<float>& x, int T, int dim, const std::vector<float>& gamma) {
    for (int t = 0; t < T; t++)
        rms_norm_inplace(x.data() + (size_t)t * dim, dim, gamma.data());
}

// Adjacent-pair RoPE (rotary_embedding_torch: theta=10000, dim=dim_head, full
// rotation) applied to one head's [T, dim_head] slice in place. inv_freq[i] =
// theta^-(2i/dim_head); pair (2i,2i+1) rotated by angle m*inv_freq[i].
void rope_head(float* qh, int T, int dim_head) {
    const int half = dim_head / 2;
    for (int m = 0; m < T; m++) {
        float* row = qh + (size_t)m * dim_head;
        for (int i = 0; i < half; i++) {
            const double inv = std::pow(10000.0, -(double)(2 * i) / (double)dim_head);
            const double ang = (double)m * inv;
            const double c = std::cos(ang), s = std::sin(ang);
            const float a = row[2 * i], b = row[2 * i + 1];
            row[2 * i] = (float)(a * c - b * s);
            row[2 * i + 1] = (float)(b * c + a * s);
        }
    }
}

// One RoFormer block (attention + FFN, both pre-RMSNorm, residual) over a
// [T, dim] sequence for a SINGLE band/batch element. Layer 0 only (is_first:
// no value-residual input). `pre` = e.g. "layers.0.0.layers.0.". Modifies x.
// #296: a block's weights, read (and F16->F32 dequantized) ONCE per layer instead
// of on every one of the num_bands / T roformer_block calls — the redundant reads
// were a large O(T) cost in run_freq.
struct RoformerBlockW {
    std::vector<float> nrm_g, qkv_w, gate_w, gate_b, out_w;
    std::vector<float> ff_g, ff1_w, ff1_b, ff4_w, ff4_b;
};
bool read_block_weights(core_gguf::WeightLoad& mw, const std::string& pre, RoformerBlockW& w) {
    return read_f32(mw, pre + "0.norm.gamma", w.nrm_g) && read_f32(mw, pre + "0.to_qkv.weight", w.qkv_w) &&
           read_f32(mw, pre + "0.to_gates.weight", w.gate_w) && read_f32(mw, pre + "0.to_gates.bias", w.gate_b) &&
           read_f32(mw, pre + "0.to_out.0.weight", w.out_w) && read_f32(mw, pre + "1.net.0.gamma", w.ff_g) &&
           read_f32(mw, pre + "1.net.1.weight", w.ff1_w) && read_f32(mw, pre + "1.net.1.bias", w.ff1_b) &&
           read_f32(mw, pre + "1.net.4.weight", w.ff4_w) && read_f32(mw, pre + "1.net.4.bias", w.ff4_b);
}

bool roformer_block(const RoformerBlockW& w, std::vector<float>& x, int T, int dim, int heads, int dim_head) {
    const int inner = heads * dim_head;
    const auto& nrm_g = w.nrm_g;
    const auto& qkv_w = w.qkv_w;
    const auto& gate_w = w.gate_w;
    const auto& gate_b = w.gate_b;
    const auto& out_w = w.out_w;
    const auto& ff_g = w.ff_g;
    const auto& ff1_w = w.ff1_w;
    const auto& ff1_b = w.ff1_b;
    const auto& ff4_w = w.ff4_w;
    const auto& ff4_b = w.ff4_b;

    // --- attention ---
    std::vector<float> xn = x;
    rms_rows(xn, T, dim, nrm_g);
    std::vector<float> qkv;
    linear(xn, T, dim, qkv_w, nullptr, inner * 3, qkv); // (T, 3*inner)
    // split into per-head q,k,v: index (qkv*heads + h)*dim_head + d
    std::vector<float> q(inner * T), k(inner * T), v(inner * T); // per head contiguous: [h][T][dh]
    for (int t = 0; t < T; t++)
        for (int h = 0; h < heads; h++)
            for (int d = 0; d < dim_head; d++) {
                const int col = h * dim_head + d;
                q[((size_t)h * T + t) * dim_head + d] = qkv[(size_t)t * inner * 3 + 0 * inner + col];
                k[((size_t)h * T + t) * dim_head + d] = qkv[(size_t)t * inner * 3 + 1 * inner + col];
                v[((size_t)h * T + t) * dim_head + d] = qkv[(size_t)t * inner * 3 + 2 * inner + col];
            }
    for (int h = 0; h < heads; h++) {
        rope_head(q.data() + (size_t)h * T * dim_head, T, dim_head);
        rope_head(k.data() + (size_t)h * T * dim_head, T, dim_head);
    }
    // attention per head, scale = dim_head^-0.5, full (no mask).
    const float scale = 1.0f / std::sqrt((float)dim_head);
    std::vector<float> attn(inner * T, 0.0f); // [h][T][dh] like q
#if defined(HAVE_BLAS)
    // #296: per head, S = scale * Q_h @ K_h^T (T x T) -> softmax rows -> O_h =
    // S @ V_h (T x dim_head). Two SGEMMs replace the scalar O(T^2*dim_head) triple
    // loop that dominated run_time; softmax stays in float (matches the reference).
    std::vector<float> S((size_t)T * T);
    for (int h = 0; h < heads; h++) {
        const float* qh = q.data() + (size_t)h * T * dim_head;
        const float* kh = k.data() + (size_t)h * T * dim_head;
        const float* vh = v.data() + (size_t)h * T * dim_head;
        float* oh = attn.data() + (size_t)h * T * dim_head;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, dim_head, scale, qh, dim_head, kh, dim_head, 0.0f,
                    S.data(), T);
        for (int m = 0; m < T; m++) {
            float* sr = S.data() + (size_t)m * T;
            float mx = -1e30f;
            for (int n = 0; n < T; n++)
                if (sr[n] > mx)
                    mx = sr[n];
            float sum = 0.0f;
            for (int n = 0; n < T; n++) {
                sr[n] = std::exp(sr[n] - mx);
                sum += sr[n];
            }
            const float inv = 1.0f / sum;
            for (int n = 0; n < T; n++)
                sr[n] *= inv;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, dim_head, T, 1.0f, S.data(), T, vh, dim_head, 0.0f,
                    oh, dim_head);
    }
#else
    std::vector<float> scores(T);
    for (int h = 0; h < heads; h++) {
        const float* qh = q.data() + (size_t)h * T * dim_head;
        const float* kh = k.data() + (size_t)h * T * dim_head;
        const float* vh = v.data() + (size_t)h * T * dim_head;
        float* oh = attn.data() + (size_t)h * T * dim_head;
        for (int m = 0; m < T; m++) {
            float mx = -1e30f;
            for (int n = 0; n < T; n++) {
                float dot = 0.0f;
                for (int d = 0; d < dim_head; d++)
                    dot += qh[(size_t)m * dim_head + d] * kh[(size_t)n * dim_head + d];
                scores[n] = dot * scale;
                if (scores[n] > mx)
                    mx = scores[n];
            }
            float sum = 0.0f;
            for (int n = 0; n < T; n++) {
                scores[n] = std::exp(scores[n] - mx);
                sum += scores[n];
            }
            for (int d = 0; d < dim_head; d++) {
                float acc = 0.0f;
                for (int n = 0; n < T; n++)
                    acc += scores[n] * vh[(size_t)n * dim_head + d];
                oh[(size_t)m * dim_head + d] = acc / sum;
            }
        }
    }
#endif
    // per-head gating: out[t,h,:] *= sigmoid(gates[t,h]); gates = xn @ gate_w.T + b
    std::vector<float> gates;
    linear(xn, T, dim, gate_w, &gate_b, heads, gates); // (T, heads)
    for (int t = 0; t < T; t++)
        for (int h = 0; h < heads; h++) {
            const double g = 1.0 / (1.0 + std::exp(-(double)gates[(size_t)t * heads + h]));
            for (int d = 0; d < dim_head; d++)
                attn[((size_t)h * T + t) * dim_head + d] *= (float)g;
        }
    // reshape [h][T][dh] -> [T][h*dh] then to_out
    std::vector<float> attn_flat((size_t)T * inner);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < heads; h++)
            for (int d = 0; d < dim_head; d++)
                attn_flat[(size_t)t * inner + h * dim_head + d] = attn[((size_t)h * T + t) * dim_head + d];
    std::vector<float> attn_out;
    linear(attn_flat, T, inner, out_w, nullptr, dim, attn_out); // (T, dim)
    for (size_t i = 0; i < x.size(); i++)
        x[i] += attn_out[i]; // residual

    // --- FFN ---
    std::vector<float> fn = x;
    rms_rows(fn, T, dim, ff_g);
    std::vector<float> h1;
    linear(fn, T, dim, ff1_w, &ff1_b, (int)ff1_b.size(), h1);
    gelu_erf_inplace(h1);
    std::vector<float> h2;
    linear(h1, T, (int)ff1_b.size(), ff4_w, &ff4_b, dim, h2);
    for (size_t i = 0; i < x.size(); i++)
        x[i] += h2[i]; // residual
    return true;
}

// Run a Transformer (one roformer_block + final RMSNorm) over the TIME axis:
// x is (T, nb, dim); each band's (T, dim) sequence is transformed independently.
bool run_time(core_gguf::WeightLoad& mw, int L, std::vector<float>& x, int T, int nb, int dim, int heads,
              int dim_head) {
    const std::string pre = "layers." + std::to_string(L) + ".0.";
    std::vector<float> fg;
    if (!read_f32(mw, pre + "norm.gamma", fg))
        return false;
    RoformerBlockW bw; // read the block weights ONCE, reuse across all bands (#296)
    if (!read_block_weights(mw, pre + "layers.0.", bw))
        return false;
    // Each band is an independent Transformer over the T-axis: distinct bands
    // write disjoint b-strides of x, and roformer_block reads only the shared
    // (const) bw, so the band loop is embarrassingly parallel. seq is thread-local.
    std::atomic<bool> ok{true};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nb; b++) {
        if (!ok.load(std::memory_order_relaxed))
            continue;
        std::vector<float> seq((size_t)T * dim);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dim; d++)
                seq[(size_t)t * dim + d] = x[((size_t)t * nb + b) * dim + d];
        if (!roformer_block(bw, seq, T, dim, heads, dim_head)) {
            ok.store(false, std::memory_order_relaxed);
            continue;
        }
        rms_rows(seq, T, dim, fg);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dim; d++)
                x[((size_t)t * nb + b) * dim + d] = seq[(size_t)t * dim + d];
    }
    return ok.load();
}

// Run a Transformer over the FREQ (band) axis: x is (T, nb, dim); each time
// step's (nb, dim) band sequence is transformed independently.
bool run_freq(core_gguf::WeightLoad& mw, int L, std::vector<float>& x, int T, int nb, int dim, int heads,
              int dim_head) {
    const std::string pre = "layers." + std::to_string(L) + ".1.";
    std::vector<float> fg;
    if (!read_f32(mw, pre + "norm.gamma", fg))
        return false;
    RoformerBlockW bw; // read the block weights ONCE, reuse across all T steps (#296)
    if (!read_block_weights(mw, pre + "layers.0.", bw))
        return false;
    // Each time step is an independent Transformer over the band-axis: distinct
    // t write disjoint nb*dim slabs of x. Parallel over T (see run_time note).
    std::atomic<bool> ok{true};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int t = 0; t < T; t++) {
        if (!ok.load(std::memory_order_relaxed))
            continue;
        std::vector<float> seq((size_t)nb * dim);
        std::memcpy(seq.data(), x.data() + (size_t)t * nb * dim, (size_t)nb * dim * sizeof(float));
        if (!roformer_block(bw, seq, nb, dim, heads, dim_head)) {
            ok.store(false, std::memory_order_relaxed);
            continue;
        }
        rms_rows(seq, nb, dim, fg);
        std::memcpy(x.data() + (size_t)t * nb * dim, seq.data(), (size_t)nb * dim * sizeof(float));
    }
    return ok.load();
}

// Mask estimator (stem 0): per band, MLP (Linear->Tanh->Linear->Tanh->Linear to
// 2*din) then GLU(-1) -> din. Concat over bands -> (T, sum(din)) = mask_raw.
// x is the post-stack (T, nb, dim).
bool mask_estimator(core_gguf::WeightLoad& mw, const std::vector<float>& x, int T, int nb, int dim,
                    const std::vector<int>& band_width, std::vector<float>& mask_raw) {
    int total = 0;
    for (int w : band_width)
        total += w;
    mask_raw.assign((size_t)T * total, 0.0f);
    std::vector<float> w0, b0, w2, b2, w4, b4, in, h;
    for (int b = 0; b < nb; b++) {
        const std::string pre = "mask_estimators.0.to_freqs." + std::to_string(b) + ".0.";
        if (!read_f32(mw, pre + "0.weight", w0) || !read_f32(mw, pre + "0.bias", b0) ||
            !read_f32(mw, pre + "2.weight", w2) || !read_f32(mw, pre + "2.bias", b2) ||
            !read_f32(mw, pre + "4.weight", w4) || !read_f32(mw, pre + "4.bias", b4))
            return false;
        const int din2 = (int)b4.size(); // 2*din
        const int hid = (int)b0.size();  // 1536
        const int din = din2 / 2;        // band width
        int off = 0;                     // column offset in mask_raw
        for (int j = 0; j < b; j++)
            off += band_width[j];
        // input = x[:, b, :] (T, dim)
        in.resize((size_t)T * dim);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dim; d++)
                in[(size_t)t * dim + d] = x[((size_t)t * nb + b) * dim + d];
        std::vector<float> a, c, e;
        linear(in, T, dim, w0, &b0, hid, a);
        for (auto& z : a)
            z = std::tanh(z);
        linear(a, T, hid, w2, &b2, hid, c);
        for (auto& z : c)
            z = std::tanh(z);
        linear(c, T, hid, w4, &b4, din2, e); // (T, 2*din)
        // GLU(-1): out[:din] * sigmoid(out[din:])
        for (int t = 0; t < T; t++) {
            const float* er = e.data() + (size_t)t * din2;
            float* mr = mask_raw.data() + (size_t)t * total + off;
            for (int i = 0; i < din; i++)
                mr[i] = er[i] * (float)(1.0 / (1.0 + std::exp(-(double)er[i + din])));
        }
    }
    return true;
}

// FFT-based inverse STFT for a power-of-2 n_fft. Numerically equivalent to
// core_istft::istft(..., TRIM_CENTER) — same Hann overlap-add, COLA window-sum
// normalization and center trim — but O(n_fft log n_fft) per frame instead of the
// shared header's naive O(n_fft^2) irfft. #296: at n_fft=2048 that DFT was ~1/3 of
// the whole separation time. Inverse via the forward FFT:
//   ifft(X) = conj(fft(conj(X)))/N,  and X is Hermitian ⇒ the output is real, so
//   x[n] = Re(fft(conj(Xfull)))[n] / N.
std::vector<float> istft_fft(const float* mag, const float* phase, int n_fft, int hop, int T_frames,
                             const float* window) {
    const int n_freq = n_fft / 2 + 1;
    const int ola_len = (T_frames - 1) * hop + n_fft;
    std::vector<float> output((size_t)ola_len, 0.0f), win_sum((size_t)ola_len, 0.0f);
    std::vector<float> re((size_t)n_fft), im((size_t)n_fft);
    const float invN = 1.0f / (float)n_fft;
    for (int t = 0; t < T_frames; t++) {
        const float* m = mag + (size_t)t * n_freq;
        const float* p = phase + (size_t)t * n_freq;
        // conj of the half spectrum (bins 0..N/2)
        for (int f = 0; f < n_freq; f++) {
            re[(size_t)f] = m[f] * std::cos(p[f]);
            im[(size_t)f] = -(m[f] * std::sin(p[f]));
        }
        // Hermitian mirror of the conjugated spectrum (bins 1..N/2-1 -> N-f)
        for (int f = 1; f < n_freq - 1; f++) {
            re[(size_t)(n_fft - f)] = re[(size_t)f];
            im[(size_t)(n_fft - f)] = -im[(size_t)f];
        }
        core_fft::fft_radix2_inplace(re.data(), im.data(), n_fft);
        const int offset = t * hop;
        for (int i = 0; i < n_fft && (offset + i) < ola_len; i++) {
            const float w = window[i];
            output[(size_t)offset + i] += re[(size_t)i] * invN * w;
            win_sum[(size_t)offset + i] += w * w;
        }
    }
    for (int i = 0; i < ola_len; i++)
        if (win_sum[(size_t)i] > 1e-8f)
            output[(size_t)i] /= win_sum[(size_t)i];
    const int pad = n_fft / 2; // TRIM_CENTER
    const int final_len = ola_len - 2 * pad;
    if (final_len <= 0)
        return {};
    return std::vector<float>(output.begin() + pad, output.begin() + pad + final_len);
}

// Apply the estimated mask to the packed STFT and iSTFT back to `channels`
// waveforms of `T_samp` samples each. `packed` is (rows=n_freqs*channels, T, 2);
// `mask_raw` is (T, 2N) complex per gather-index. Shared by separate() and the
// diff's output_vocals stage.
void synthesize(mel_band_roformer_context* ctx, const std::vector<float>& packed, const std::vector<float>& mask_raw,
                int T, int T_samp, std::vector<float>& out /* channels*T_samp */) {
    const int channels = ctx->hp.audio_channels;
    const int n_freqs = ctx->n_freqs();
    const int rows = n_freqs * channels;
    const int N = (int)ctx->freq_indices.size();

    std::vector<double> msum_re((size_t)rows * T, 0.0), msum_im((size_t)rows * T, 0.0);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < N; k++) {
            const int r = ctx->freq_indices[k];
            msum_re[(size_t)r * T + t] += mask_raw[(size_t)t * (2 * N) + 2 * k + 0];
            msum_im[(size_t)r * T + t] += mask_raw[(size_t)t * (2 * N) + 2 * k + 1];
        }
    std::vector<float> mspec_re((size_t)rows * T), mspec_im((size_t)rows * T);
    for (int r = 0; r < rows; r++) {
        const double denom = std::max((double)ctx->num_bands_per_freq[r / channels], 1e-8);
        for (int t = 0; t < T; t++) {
            const double mr = msum_re[(size_t)r * T + t] / denom, mi = msum_im[(size_t)r * T + t] / denom;
            const float sr = packed[((size_t)r * T + t) * 2 + 0], si = packed[((size_t)r * T + t) * 2 + 1];
            mspec_re[(size_t)r * T + t] = (float)(sr * mr - si * mi);
            mspec_im[(size_t)r * T + t] = (float)(sr * mi + si * mr);
        }
    }
    out.assign((size_t)channels * T_samp, 0.0f);
    std::vector<float> mag((size_t)T * n_freqs), phase((size_t)T * n_freqs), win(ctx->hp.n_fft);
    core_istft::hann_periodic(ctx->hp.n_fft, win.data());
    for (int s = 0; s < channels; s++) {
        for (int t = 0; t < T; t++)
            for (int f = 0; f < n_freqs; f++) {
                const int r = f * channels + s;
                const float re = mspec_re[(size_t)r * T + t], im = mspec_im[(size_t)r * T + t];
                mag[(size_t)t * n_freqs + f] = std::sqrt(re * re + im * im);
                phase[(size_t)t * n_freqs + f] = std::atan2(im, re);
            }
        const bool pow2 = ctx->hp.n_fft > 0 && (ctx->hp.n_fft & (ctx->hp.n_fft - 1)) == 0;
        std::vector<float> wav = pow2 ? istft_fft(mag.data(), phase.data(), ctx->hp.n_fft, ctx->hp.hop, T, win.data())
                                      : core_istft::istft(mag.data(), phase.data(), ctx->hp.n_fft, ctx->hp.hop, T,
                                                          win.data(), core_istft::TRIM_CENTER);
        for (int i = 0; i < T_samp && i < (int)wav.size(); i++)
            out[(size_t)s * T_samp + i] = wav[i];
    }
}

// Full validated forward on real per-channel PCM (chan[s] has T_samp samples).
// Produces the vocal stem interleaved (channels-interleaved, T_samp frames).
bool run_forward(mel_band_roformer_context* ctx, const std::vector<std::vector<float>>& chan, int T_samp,
                 std::vector<float>& vocals_interleaved) {
    const auto& hp = ctx->hp;
    const int channels = hp.audio_channels, dim = hp.dim, nb = hp.num_bands;
    const int n_freqs = ctx->n_freqs();
    const int T = stft_n_frames(T_samp, hp.hop);

#if defined(CRISPASR_MBR_OPENBLAS)
    // Coarse OpenMP parallelism (band/time loops) does the threading; keep each
    // per-block OpenBLAS call serial so they don't oversubscribe cores.
    openblas_set_num_threads(1);
#endif

    // #296: per-stage profiling (CRISPASR_MBR_PROFILE=1) to localise where the
    // forward spends time — the separation was silently slow and the bottleneck
    // was not where it looked.
    const bool prof = std::getenv("CRISPASR_MBR_PROFILE") != nullptr;
    using clk = std::chrono::steady_clock;
    auto tick = clk::now();
    auto lap = [&](const char* what) {
        if (prof) {
            auto n = clk::now();
            fprintf(stderr, "  [mbr-prof] %-14s %7lld ms\n", what,
                    (long long)std::chrono::duration_cast<std::chrono::milliseconds>(n - tick).count());
            tick = n;
        }
    };
    std::vector<float> window;
    hann_periodic(hp.win, window);
    std::vector<std::vector<float>> chan_spec(channels);
    for (int s = 0; s < channels; s++)
        stft_one_channel(chan[s].data(), T_samp, hp.n_fft, hp.hop, window, T, n_freqs, chan_spec[s]);
    std::vector<float> packed;
    pack_stft(chan_spec, n_freqs, T, channels, packed);
    lap("stft+pack");
    std::vector<float> gathered;
    band_gather(packed, ctx->freq_indices, T, gathered);
    std::vector<float> x;
    bool bs_ok;
    if (ctx->use_graph) {
        // Change 176 (Phase 1): ggml-graph band-split on ctx->backend. Parity
        // gate: output layout identical to band_split_cpu, compared in the
        // diff harness (band_split_out stage).
        bs_ok = band_split_graph(ctx, gathered, T, x);
        lap("band_split(graph)");
    } else {
        bs_ok = band_split_cpu(ctx->weights, gathered, ctx->band_width, T, dim, x);
        lap("band_split");
    }
    if (!bs_ok)
        return false;
    // Compute-heavy Transformer stack; emit per-layer progress so it never looks
    // hung, and (under profiling) split run_time vs run_freq time.
    fprintf(stderr, "mel_band_roformer: separating (T=%d frames, %d layers, %d bands)...\n", T, hp.depth, nb);
    double t_time = 0, t_freq = 0;
    for (int L = 0; L < hp.depth; L++) {
        fprintf(stderr, "mel_band_roformer: layer %d/%d\n", L + 1, hp.depth);
        auto a = clk::now();
        if (!run_time(ctx->weights, L, x, T, nb, dim, hp.heads, hp.dim_head))
            return false;
        auto b = clk::now();
        if (!run_freq(ctx->weights, L, x, T, nb, dim, hp.heads, hp.dim_head))
            return false;
        auto c = clk::now();
        t_time += std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        t_freq += std::chrono::duration_cast<std::chrono::milliseconds>(c - b).count();
    }
    if (prof)
        fprintf(stderr, "  [mbr-prof] run_time(all)  %7.0f ms\n  [mbr-prof] run_freq(all)  %7.0f ms\n", t_time, t_freq);
    tick = clk::now();
    std::vector<float> mask_raw;
    if (!mask_estimator(ctx->weights, x, T, nb, dim, ctx->band_width, mask_raw))
        return false;
    lap("mask_est");
    std::vector<float> out_planar; // channels * T_samp
    synthesize(ctx, packed, mask_raw, T, T_samp, out_planar);
    lap("synthesize");
    // interleave
    vocals_interleaved.assign((size_t)T_samp * channels, 0.0f);
    for (int i = 0; i < T_samp; i++)
        for (int s = 0; s < channels; s++)
            vocals_interleaved[(size_t)i * channels + s] = out_planar[(size_t)s * T_samp + i];
    return true;
}

} // namespace

mel_band_roformer_result* mel_band_roformer_separate(mel_band_roformer_context* ctx, const float* pcm, int n_samples,
                                                     int in_channels) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;
    const int channels = ctx->hp.audio_channels;
    // De-interleave input, up/down-mixing to the model's channel count.
    std::vector<std::vector<float>> chan(channels, std::vector<float>(n_samples, 0.0f));
    for (int i = 0; i < n_samples; i++)
        for (int s = 0; s < channels; s++) {
            int src = (in_channels <= 0) ? 0 : (s < in_channels ? s : in_channels - 1);
            chan[s][i] = pcm[(size_t)i * (in_channels > 0 ? in_channels : 1) + src];
        }

    std::vector<float> vocals; // interleaved
    if (!run_forward(ctx, chan, n_samples, vocals))
        return nullptr;

    // Stem 0 = vocals (model output). Stem 1 = other = input - vocals (residual).
    const int n_sources = 2;
    auto* r = (mel_band_roformer_result*)calloc(1, sizeof(mel_band_roformer_result));
    r->n_sources = n_sources;
    r->n_channels = channels;
    r->n_samples = n_samples;
    r->sample_rate = ctx->hp.sample_rate;
    r->sources = (float**)calloc(n_sources, sizeof(float*));
    r->source_names = (const char**)calloc(n_sources, sizeof(char*));
    const size_t nf = (size_t)n_samples * channels;
    r->sources[0] = (float*)malloc(nf * sizeof(float));
    r->sources[1] = (float*)malloc(nf * sizeof(float));
    for (size_t i = 0; i < nf; i++)
        r->sources[0][i] = vocals[i];
    // residual = original (interleaved, mixed to model channels) - vocals
    for (int i = 0; i < n_samples; i++)
        for (int s = 0; s < channels; s++) {
            const size_t idx = (size_t)i * channels + s;
            r->sources[1][idx] = chan[s][i] - vocals[idx];
        }
    r->source_names[0] = ctx->source_names_storage[0].c_str();
    r->source_names[1] = ctx->source_names_storage.size() > 1 ? ctx->source_names_storage[1].c_str() : "other";
    return r;
}

int mel_band_roformer_parity(const char* model_gguf, const char* audio_wav, int verbosity) {
    // Change 176 Phase 1: standalone parity — ggml-graph band-split vs the
    // validated CPU reference on IDENTICAL input (the same wav, run through the
    // same STFT/gather front-end). The CPU path is cos=1.0 vs the Python
    // fixture, so graph==CPU proves the cleanroom port without needing torch or
    // the fixture. Returns 0 iff the graph stage passes the diff threshold.
    mel_band_roformer_context* ctx = mel_band_roformer_init_from_file(model_gguf, mel_band_roformer_default_params());
    if (!ctx) {
        fprintf(stderr, "mbr_parity: failed to load model %s\n", model_gguf);
        return 2;
    }

    std::vector<float> pcm;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(audio_wav, pcm, sr)) {
        fprintf(stderr, "mbr_parity: failed to read wav %s\n", audio_wav);
        mel_band_roformer_free(ctx);
        return 2;
    }
    // The front-end expects `channels` per-channel arrays; mono wav -> 1 channel.
    const int channels = ctx->hp.audio_channels;
    std::vector<std::vector<float>> chan(channels, std::vector<float>(pcm.size(), 0.0f));
    for (size_t i = 0; i < pcm.size(); i++)
        for (int s = 0; s < channels; s++)
            chan[s][i] = pcm[i];

    const auto& hp = ctx->hp;
    const int n_freqs = ctx->n_freqs();
    const int T_samp = (int)pcm.size();
    const int T = stft_n_frames(T_samp, hp.hop);

    std::vector<float> window;
    hann_periodic(hp.win, window);
    std::vector<std::vector<float>> chan_spec(channels);
    for (int s = 0; s < channels; s++)
        stft_one_channel(chan[s].data(), T_samp, hp.n_fft, hp.hop, window, T, n_freqs, chan_spec[s]);
    std::vector<float> packed;
    pack_stft(chan_spec, n_freqs, T, channels, packed);
    std::vector<float> gathered;
    band_gather(packed, ctx->freq_indices, T, gathered);

    fprintf(stderr, "mel_band_roformer parity (T_samp=%d, T=%d, n_freqs=%d, channels=%d, N_gather=%zu):\n", T_samp, T,
            n_freqs, channels, ctx->freq_indices.size());

    std::vector<float> cpu_out, graph_out;
    const bool cpu_ok = band_split_cpu(ctx->weights, gathered, ctx->band_width, T, hp.dim, cpu_out);
    const bool graph_ok = band_split_graph(ctx, gathered, T, graph_out);
    if (!cpu_ok || !graph_ok) {
        fprintf(stderr, "mbr_parity: band-split failed (cpu=%d graph=%d)\n", cpu_ok ? 1 : 0, graph_ok ? 1 : 0);
        mel_band_roformer_free(ctx);
        return 2;
    }

    const int64_t n = (int64_t)std::min(cpu_out.size(), graph_out.size());
    const double cos = cosine(graph_out.data(), cpu_out.data(), n);
    const double mad = max_abs_diff(graph_out.data(), cpu_out.data(), n);
    const double lg = l2_norm(graph_out.data(), n), lc = l2_norm(cpu_out.data(), n);
    const bool ok = cos >= 0.9995 && graph_out.size() == cpu_out.size();
    if (verbosity >= 1 || !ok) {
        fprintf(stderr, "  %-16s %s cos=%.6f max_abs=%.3e  (graph=%zu cpu=%zu)\n", "band_split_graph", ok ? "PASS" : "FAIL",
                cos, mad, graph_out.size(), cpu_out.size());
        if (verbosity >= 2)
            fprintf(stderr, "    |graph|=%.6f |cpu|=%.6f\n", lg, lc);
    }
    mel_band_roformer_free(ctx);
    fprintf(stderr, "mel_band_roformer parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int mel_band_roformer_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {

    mel_band_roformer_context* ctx = mel_band_roformer_init_from_file(model_gguf, mel_band_roformer_default_params());
    if (!ctx) {
        fprintf(stderr, "mbr_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "mbr_ref", rw)) {
        fprintf(stderr, "mbr_diff: failed to load reference %s\n", ref_gguf);
        mel_band_roformer_free(ctx);
        return 2;
    }

    const auto& hp = ctx->hp;
    const int channels = hp.audio_channels;
    const int n_freqs = ctx->n_freqs();

    int n_fail = 0;
    const double COS_MIN = 0.9995;

    auto report = [&](const char* stage, const std::vector<float>& mine, const std::vector<float>& ref) {
        const int64_t n = (int64_t)std::min(mine.size(), ref.size());
        const double cos = cosine(mine.data(), ref.data(), n);
        const double mad = max_abs_diff(mine.data(), ref.data(), n);
        const bool ok = cos >= COS_MIN && mine.size() == ref.size();
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok) {
            fprintf(stderr, "  %-16s %s cos=%.6f max_abs=%.3e  (mine=%zu ref=%zu)%s\n", stage, ok ? "PASS" : "FAIL",
                    cos, mad, mine.size(), ref.size(),
                    verbosity >= 2 ? ("  |mine|=" + std::to_string(l2_norm(mine.data(), n)) +
                                      " |ref|=" + std::to_string(l2_norm(ref.data(), n)))
                                         .c_str()
                                   : "");
        }
    };

    // --- input_audio (2, T_samp), row-major s-major ---
    std::vector<float> in_audio;
    int64_t in_n = 0;
    if (!ref_get(rw, "input_audio", in_audio, in_n)) {
        fprintf(stderr, "mbr_diff: reference has no input_audio stage — re-dump with the updated dumper\n");
        mel_band_roformer_free(ctx);
        return 2;
    }
    const int T_samp = (int)(in_n / channels);
    const int T = stft_n_frames(T_samp, hp.hop);

    // --- run our STFT front-end on the reference input ---
    std::vector<float> window;
    hann_periodic(hp.win, window);
    std::vector<std::vector<float>> chan_spec(channels);
    for (int s = 0; s < channels; s++)
        stft_one_channel(in_audio.data() + (size_t)s * T_samp, T_samp, hp.n_fft, hp.hop, window, T, n_freqs,
                         chan_spec[s]);

    std::vector<float> packed;
    pack_stft(chan_spec, n_freqs, T, channels, packed);
    std::vector<float> gathered;
    band_gather(packed, ctx->freq_indices, T, gathered);

    fprintf(stderr, "mel_band_roformer diff (T_samp=%d, T=%d, n_freqs=%d, channels=%d, N_gather=%zu):\n", T_samp, T,
            n_freqs, channels, ctx->freq_indices.size());

    // --- freq_indices (integer membership) as float compare ---
    {
        std::vector<float> mine_fi(ctx->freq_indices.begin(), ctx->freq_indices.end());
        std::vector<float> ref_fi;
        int64_t nn = 0;
        if (ref_get(rw, "freq_indices", ref_fi, nn))
            report("freq_indices", mine_fi, ref_fi);
    }
    // --- num_bands_per_freq (overlap denominator) as float compare ---
    // Structural, like freq_indices, and the reference dumper calls out why it
    // matters: it is the denominator used when scattering bands back to the
    // spectrogram, so an off-by-one silently corrupts every downstream band.
    // It was captured by the dumper but never compared here.
    {
        std::vector<float> mine_nb(ctx->num_bands_per_freq.begin(), ctx->num_bands_per_freq.end());
        std::vector<float> ref_nb;
        int64_t nn = 0;
        if (ref_get(rw, "num_bands_per_freq", ref_nb, nn))
            report("num_bands_per_freq", mine_nb, ref_nb);
    }
    // --- stft_packed (f*s, T, 2) ---
    {
        std::vector<float> ref_sp;
        int64_t nn = 0;
        if (ref_get(rw, "stft_packed", ref_sp, nn))
            report("stft_packed", packed, ref_sp);
    }
    // --- band_gathered (T, N*2) ---
    {
        std::vector<float> ref_bg;
        int64_t nn = 0;
        if (ref_get(rw, "band_gathered", ref_bg, nn))
            report("band_gathered", gathered, ref_bg);
    }
    // --- band_split_out (T, num_bands, dim) ---
    // INPUT-ALIGNED: fed the REFERENCE band_gathered, not our STFT output. The
    // band_split RMSNorm divides by each band's norm, so it amplifies the ~5e-4
    // float-level difference between our FFT and torch's on near-silent bands
    // (a uniform ±5e-4 perturbation alone drops this cos to ~0.73). Diffing off
    // the ref input isolates the band_split MATH; the STFT is validated
    // separately by the stft_packed stage. (Diff-harness rule: gate input
    // alignment before trusting a per-layer cos.)
    {
        std::vector<float> ref_bg_in;
        int64_t nn = 0;
        if (ref_get(rw, "band_gathered", ref_bg_in, nn)) {
            std::vector<float> bso;
            if (band_split_cpu(ctx->weights, ref_bg_in, ctx->band_width, T, hp.dim, bso)) {
                std::vector<float> ref_bso;
                int64_t mm = 0;
                if (ref_get(rw, "band_split_out", ref_bso, mm))
                    report("band_split_out", bso, ref_bso);
            } else {
                fprintf(stderr, "  band_split_out: SKIP (a band weight was missing)\n");
            }
        }
    }

    // --- band_split_out (GRAPH): Change 176 Phase 1 parity gate ---
    // The ggml-graph band-split vs the validated CPU reference, on IDENTICAL
    // input (the same reference band_gathered). This is the Phase 1 acceptance
    // test: the CPU path is already cos=1.0 vs the Python fixture, so
    // graph==CPU proves the cleanroom port. Uses a fresh default ctx so the
    // graph runs on the same backend as the harness ctx (no flag pollution).
    {
        std::vector<float> ref_bg_in;
        int64_t nn = 0;
        if (ref_get(rw, "band_gathered", ref_bg_in, nn)) {
            std::vector<float> cpu_out, graph_out;
            if (band_split_cpu(ctx->weights, ref_bg_in, ctx->band_width, T, hp.dim, cpu_out) &&
                band_split_graph(ctx, ref_bg_in, T, graph_out)) {
                report("band_split_graph", graph_out, cpu_out);
            } else {
                fprintf(stderr, "  band_split_graph: SKIP (graph or CPU band-split failed)\n");
            }
        }
    }

    // --- layer0_time (band 0) : the time RoFormer block of layer 0 ---
    // INPUT-ALIGNED off the reference band_split_out (its RMSNorm amplifies
    // upstream float error). The dumper hook captured [0] = band 0's output, and
    // the time transformer attends over T within each band independently, so we
    // reproduce band 0's (T, dim) sequence only. Layer 0 is is_first -> no
    // value-residual input. Block = attn+ffn (depth 1) then the final RMSNorm.
    {
        std::vector<float> ref_bso;
        int64_t nn = 0;
        std::vector<float> ref_lt;
        int64_t mm = 0;
        if (ref_get(rw, "band_split_out", ref_bso, nn) && ref_get(rw, "layer0_time", ref_lt, mm)) {
            const int nb = hp.num_bands;
            // band 0 sequence: band_split_out[t, 0, :] -> (T, dim)
            std::vector<float> x0((size_t)T * hp.dim);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < hp.dim; d++)
                    x0[(size_t)t * hp.dim + d] = ref_bso[((size_t)t * nb + 0) * hp.dim + d];
            std::vector<float> final_g;
            RoformerBlockW tbw;
            if (read_block_weights(ctx->weights, "layers.0.0.layers.0.", tbw) &&
                roformer_block(tbw, x0, T, hp.dim, hp.heads, hp.dim_head) &&
                read_f32(ctx->weights, "layers.0.0.norm.gamma", final_g)) {
                rms_rows(x0, T, hp.dim, final_g); // Transformer final norm
                report("layer0_time", x0, ref_lt);
            } else {
                fprintf(stderr, "  layer0_time: SKIP (a weight was missing)\n");
            }
        }
    }

    // --- layer0_freq (time step 0) : the freq RoFormer block of layer 0 ---
    // The freq transformer attends over the 60 bands per time step (RoPE on band
    // positions). Its input is the FULL time-transformer output; the dumper hook
    // captured only band 0 of layer0_time, so here we CHAIN our own (validated)
    // time block over all bands and take the t=0 band-sequence as the freq
    // input. Time outputs aren't near-silent, so RMSNorm amplification is mild;
    // if this ever drifts, dump the full layer0_time and input-align instead.
    {
        std::vector<float> ref_bso, ref_lf, tfinal_g;
        int64_t n1 = 0, n2 = 0;
        if (ref_get(rw, "band_split_out", ref_bso, n1) && ref_get(rw, "layer0_freq", ref_lf, n2) &&
            read_f32(ctx->weights, "layers.0.0.norm.gamma", tfinal_g)) {
            const int nb = hp.num_bands, dim = hp.dim;
            // freq input at t=0: run the time block on each band, take t=0.
            std::vector<float> freq_in((size_t)nb * dim, 0.0f);
            RoformerBlockW tbw;
            bool ok = read_block_weights(ctx->weights, "layers.0.0.layers.0.", tbw);
            for (int b = 0; b < nb && ok; b++) {
                std::vector<float> xb((size_t)T * dim);
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < dim; d++)
                        xb[(size_t)t * dim + d] = ref_bso[((size_t)t * nb + b) * dim + d];
                ok = roformer_block(tbw, xb, T, dim, hp.heads, hp.dim_head);
                if (!ok)
                    break;
                rms_rows(xb, T, dim, tfinal_g);
                for (int d = 0; d < dim; d++)
                    freq_in[(size_t)b * dim + d] = xb[(size_t)0 * dim + d]; // t=0
            }
            std::vector<float> ffinal_g;
            RoformerBlockW fbw;
            if (ok && read_block_weights(ctx->weights, "layers.0.1.layers.0.", fbw) &&
                roformer_block(fbw, freq_in, nb, dim, hp.heads, hp.dim_head) &&
                read_f32(ctx->weights, "layers.0.1.norm.gamma", ffinal_g)) {
                rms_rows(freq_in, nb, dim, ffinal_g);
                report("layer0_freq(chain)", freq_in, ref_lf);
            } else {
                fprintf(stderr, "  layer0_freq: SKIP (a weight was missing)\n");
            }
        }
    }

    // --- mask_raw : full 6-layer stack + mask estimator (the whole learned
    // forward), input-aligned off the reference band_split_out. Validates the
    // entire transformer stack (no value residuals in 0.3.10) + mask MLP+GLU. ---
    {
        std::vector<float> x, ref_mr;
        int64_t n1 = 0, n2 = 0;
        if (ref_get(rw, "band_split_out", x, n1) && ref_get(rw, "mask_raw", ref_mr, n2)) {
            const int nb = hp.num_bands, dim = hp.dim;
            bool ok = true;
            for (int L = 0; L < hp.depth && ok; L++) {
                ok = run_time(ctx->weights, L, x, T, nb, dim, hp.heads, hp.dim_head) &&
                     run_freq(ctx->weights, L, x, T, nb, dim, hp.heads, hp.dim_head);
            }
            std::vector<float> mr;
            if (ok && mask_estimator(ctx->weights, x, T, nb, dim, ctx->band_width, mr))
                report("mask_raw", mr, ref_mr);
            else
                fprintf(stderr, "  mask_raw: SKIP (a weight was missing)\n");
        }
    }

    // --- output_vocals : scatter-average + complex mask + iSTFT ---
    // Input-aligned off the REFERENCE stft_packed + mask_raw, so this tests only
    // the DSP tail. mask_raw (T, 2N) is complex per gather-index k; scatter-add
    // into the full packed spectrum at freq_indices (summing overlapping bands),
    // divide by num_bands_per_freq, complex-multiply the stft, split channels,
    // iSTFT (torch center=True). Stem 0 = vocals.
    {
        std::vector<float> ref_sp, ref_mr, ref_ov;
        int64_t n1 = 0, n2 = 0, n3 = 0;
        if (ref_get(rw, "stft_packed", ref_sp, n1) && ref_get(rw, "mask_raw", ref_mr, n2) &&
            ref_get(rw, "output_vocals", ref_ov, n3)) {
            const int rows = n_freqs * channels; // 2050
            const int N = (int)ctx->freq_indices.size();
            // scatter-average complex mask into the full packed rows.
            std::vector<double> msum_re((size_t)rows * T, 0.0), msum_im((size_t)rows * T, 0.0);
            for (int t = 0; t < T; t++)
                for (int k = 0; k < N; k++) {
                    const int r = ctx->freq_indices[k];
                    msum_re[(size_t)r * T + t] += ref_mr[(size_t)t * (2 * N) + 2 * k + 0];
                    msum_im[(size_t)r * T + t] += ref_mr[(size_t)t * (2 * N) + 2 * k + 1];
                }
            // complex-multiply stft_packed by the averaged mask, per row/time.
            std::vector<float> mspec_re((size_t)rows * T), mspec_im((size_t)rows * T);
            for (int r = 0; r < rows; r++) {
                const double denom = std::max((double)ctx->num_bands_per_freq[r / channels], 1e-8);
                for (int t = 0; t < T; t++) {
                    const double mr_re = msum_re[(size_t)r * T + t] / denom;
                    const double mr_im = msum_im[(size_t)r * T + t] / denom;
                    const float sr = ref_sp[((size_t)r * T + t) * 2 + 0];
                    const float si = ref_sp[((size_t)r * T + t) * 2 + 1];
                    mspec_re[(size_t)r * T + t] = (float)(sr * mr_re - si * mr_im);
                    mspec_im[(size_t)r * T + t] = (float)(sr * mr_im + si * mr_re);
                }
            }
            // per channel: build (T, n_freqs) mag/phase, iSTFT.
            std::vector<float> out((size_t)channels * T_samp, 0.0f);
            std::vector<float> mag((size_t)T * n_freqs), phase((size_t)T * n_freqs), win;
            win.resize(hp.n_fft);
            core_istft::hann_periodic(hp.n_fft, win.data());
            for (int s = 0; s < channels; s++) {
                for (int t = 0; t < T; t++)
                    for (int f = 0; f < n_freqs; f++) {
                        const int r = f * channels + s;
                        const float re = mspec_re[(size_t)r * T + t], im = mspec_im[(size_t)r * T + t];
                        mag[(size_t)t * n_freqs + f] = std::sqrt(re * re + im * im);
                        phase[(size_t)t * n_freqs + f] = std::atan2(im, re);
                    }
                std::vector<float> wav = core_istft::istft(mag.data(), phase.data(), hp.n_fft, hp.hop, T, win.data(),
                                                           core_istft::TRIM_CENTER);
                for (int i = 0; i < T_samp && i < (int)wav.size(); i++)
                    out[(size_t)s * T_samp + i] = wav[i];
            }
            report("output_vocals", out, ref_ov);
        }
    }


    // Declare the coverage gap rather than letting a screen of PASS lines imply
    // full coverage. The dumper captures layer1_* and layer5_*; comparing them
    // would mean chaining our own time blocks through five layers (the
    // layer0_freq path already does this for one), which buys localisation but
    // not detection -- output_vocals is compared end to end, so a regression in
    // any intermediate layer still fails the run. Listed so the next person
    // knows where a failure would NOT be pinpointed.
    {
        const char* uncompared[] = {"layer1_time", "layer1_freq", "layer5_time", "layer5_freq"};
        std::string present;
        for (const char* nm : uncompared) {
            std::vector<float> tmp;
            int64_t nn = 0;
            if (ref_get(rw, nm, tmp, nn))
                present += (present.empty() ? "" : ", ") + std::string(nm);
        }
        if (!present.empty())
            fprintf(stderr,
                    "  NOTE: in the reference but not compared: %s\n"
                    "        (end-to-end output_vocals IS compared, so a regression there still\n"
                    "         fails -- these would only localise it to a layer)\n",
                    present.c_str());
    }

    if (rw.buf)
        core_gguf::release_weight_buffer(rw.buf);
    if (rw.ctx)
        ggml_free(rw.ctx);
    mel_band_roformer_free(ctx);
    fprintf(stderr, "mel_band_roformer diff: %d front-end stage(s) FAILED.\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
