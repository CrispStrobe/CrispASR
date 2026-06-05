// mimo_tokenizer.cpp — MiMo-Audio-Tokenizer encoder runtime.
//
// PLAN #51 step 2: full forward path.
//
//   16 kHz PCM
//     → polyphase sinc resample (16k → 24k)
//     → log-mel (n_fft=960, hop=240, n_mels=128, magnitude, log-clip 1e-7)
//     → conv1 (Conv1d 128→1280, k=3, p=1) + GELU
//     → conv2 (Conv1d 1280→1280, k=3, s=2, p=1) + GELU
//     → 32-layer pre-LN transformer (RoPE θ=10000, head_dim=64,
//                                    q/v/o have biases, k has none,
//                                    GELU FFN 1280→5120, skip-add from
//                                    output of layer 2 added back after
//                                    output of final layer)
//     → final LayerNorm
//     → down_sample (Conv1d 1280→1280, k=2, s=2, no bias) + GELU
//     → down_sample_norm (LayerNorm)
//     → 8-stage Euclidean RVQ encode (CPU-side argmin)
//
// Architecture is documented in src/mimo_tokenizer.h and mirrored in the
// Python reference dumper at tools/reference_backends/mimo_tokenizer.py.
//
// Notes:
// - The graph is CPU-pinned by default for the encoder (the conv stem is
//   the same shape that hit the M1 `kernel_conv_transpose_1d` watchdog
//   on qwen3-tts; this is forward conv, not transpose, but we start
//   conservative until validated against the Python ref dumper).
// - The 16 → 24 kHz resampler is a simple polyphase sinc (Hann window,
//   lowpass_filter_width=6) approximating torchaudio's default
//   resampling_method="sinc_interp_hann". It will not be bit-equivalent
//   to torchaudio, so cos against the Python dump is best ≥0.99 in early
//   stages until we tighten the resampler.

#include "mimo_tokenizer.h"

#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"
#include "core/rvq.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

struct mimo_tok_hp {
    uint32_t d_model = 1280;
    uint32_t encoder_layers = 32;
    uint32_t encoder_heads = 20;
    uint32_t encoder_ffn_dim = 5120;
    uint32_t num_quantizers = 20; // total stages on disk; ASR uses first 8
    uint32_t sampling_rate = 24000;
    uint32_t hop_length = 240;
    uint32_t stride_size = 2; // conv2 stride
    uint32_t avg_pooler = 2;  // down_sample stride
    uint32_t kernel_size = 3; // conv stem kernel
    // Defaults below come from MiMo-Audio-Tokenizer/config.json — they are
    // not yet written into the GGUF by the converter, so we baked them in.
    uint32_t encoder_skip_layer_id = 3; // skip-connection saved AFTER layer 2
    uint32_t n_mels = 128;
    uint32_t nfft = 960;
    uint32_t window_size = 960;
    float rope_theta = 10000.0f;
    // Per-stage codebook sizes (1024,1024,128×6 for the first 8 ASR stages).
    std::vector<uint32_t> codebook_size;
};

struct mimo_tok_layer {
    // LayerNorm (pre-attn): self_attn_layer_norm
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    // Attention: q/v/o have biases, k has NO bias (upstream-defined).
    ggml_tensor* q_w = nullptr;
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr; // no bias
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* o_b = nullptr;
    // LayerNorm (pre-FFN): final_layer_norm
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
    // FFN: fc1 (d → ffn) + GELU + fc2 (ffn → d), both with biases.
    ggml_tensor* fc1_w = nullptr;
    ggml_tensor* fc1_b = nullptr;
    ggml_tensor* fc2_w = nullptr;
    ggml_tensor* fc2_b = nullptr;
};

struct mimo_tok_codebook {
    // Per-stage: F16 [d_model, codebook_size] in GGUF on-disk order
    // (innermost dim = d_model). Loaded as F16 then promoted to F32 at
    // distance-compute time (matches `quantizer.float()` in upstream).
    ggml_tensor* embed = nullptr;
    uint32_t codebook_size = 0;
    // Cached F32 copy of the embedding rows in row-major (cb_size, d_model)
    // layout for fast Euclidean argmin. Populated on first encode() call.
    std::vector<float> embed_f32;
    // Cached ||embed[k]||^2 per row for the argmin shortcut.
    std::vector<float> embed_norm_sq;
};

constexpr float kLayerNormEps = 1e-5f;

} // namespace

struct mimo_tokenizer_context {
    mimo_tokenizer_context_params params{};
    int n_threads = 4;

    mimo_tok_hp hp;

    // Backends + weights.
    ggml_backend_t backend = nullptr;     // best-available (Metal/GPU or CPU)
    ggml_backend_t backend_cpu = nullptr; // always present
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Bound tensor handles (encoder forward path).
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* final_norm_b = nullptr;
    ggml_tensor* down_w = nullptr; // Conv1d (1280, 1280, k=2, s=2, no bias)
    ggml_tensor* down_norm_w = nullptr;
    ggml_tensor* down_norm_b = nullptr;
    std::vector<mimo_tok_layer> layers;
    std::vector<mimo_tok_codebook> codebooks; // first 8 used for ASR
};

// ===========================================================================
// 16 → 24 kHz polyphase sinc resampler (best-effort match to torchaudio's
// `resampling_method="sinc_interp_hann"` default).
// ===========================================================================

namespace {

static int gcd_i(int a, int b) {
    while (b) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static std::vector<float> resample_sinc_hann(const float* x, int n_in, int sr_in, int sr_out) {
    if (sr_in == sr_out)
        return std::vector<float>(x, x + n_in);
    const int g = gcd_i(sr_in, sr_out);
    const int up = sr_out / g;
    const int down = sr_in / g;
    // Shared upsampled rate
    const int sr_up = sr_in * up; // == sr_out * down

    // torchaudio: lowpass cutoff = min(orig, new) * rolloff (rolloff=0.99)
    const float rolloff = 0.99f;
    const float fc_hz = std::min(sr_in, sr_out) * 0.5f * rolloff;
    // Filter half-width in upsampled-rate samples (lowpass_filter_width=6 zero
    // crossings of the sinc at the lower of the two rates).
    const int lpw = 6;
    const int half = lpw * std::max(up, down); // taps = 2*half + 1
    const int taps = 2 * half + 1;

    // Build filter: h[k] = sinc(2 * fc_hz / sr_up * (k - half)) * hann
    std::vector<float> h(taps);
    const float fc_norm = fc_hz / (0.5f * (float)sr_up); // in (0, 1]
    double sum = 0.0;
    for (int k = 0; k < taps; k++) {
        int n = k - half;
        float arg = (float)M_PI * fc_norm * (float)n;
        float sinc = (n == 0) ? 1.0f : (std::sin(arg) / arg);
        // Hann window over the filter
        float w = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)k / (float)(taps - 1));
        h[k] = fc_norm * sinc * w;
        sum += h[k];
    }
    // Normalize so DC gain is `up` (so downsample-by-`down` gives unit DC gain).
    if (sum > 0.0) {
        const float scale = (float)up / (float)sum;
        for (float& v : h)
            v *= scale;
    }

    // Output length: floor(n_in * up / down) approximately (torchaudio uses
    // ceil(n_in * up / down) — matching that here).
    const int n_up = n_in * up;
    const int n_out = (n_up + down - 1) / down;
    std::vector<float> out((size_t)n_out, 0.0f);

    // Direct convolution: for each output sample, m corresponds to
    // upsampled position p = m * down. We sum h[k] * x_up[p + k - half]
    // where x_up[i] = (i % up == 0) ? x[i / up] : 0.
    // Skipping zeros: only k where (p + k - half) mod up == 0 contribute,
    // and the corresponding input index is (p + k - half) / up.
    for (int m = 0; m < n_out; m++) {
        const int p = m * down;
        // start k so that (p + k - half) >= 0  =>  k >= half - p
        // also (p + k - half) <= n_up - 1   =>  k <= half - p + n_up - 1
        // and (p + k - half) % up == 0      =>  k ≡ (half - p) (mod up)
        int k_lo = half - p;
        if (k_lo < 0)
            k_lo = 0;
        int k_hi = half - p + (n_in * up) - 1;
        if (k_hi > taps - 1)
            k_hi = taps - 1;
        // Find the smallest k >= k_lo such that (p + k - half) % up == 0.
        int phase = ((p - half) % up + up) % up; // (i mod up) at k=0
        int k_start = k_lo;
        int phase_at_k = ((phase + k_lo) % up + up) % up;
        if (phase_at_k != 0)
            k_start += (up - phase_at_k);
        double acc = 0.0;
        for (int k = k_start; k <= k_hi; k += up) {
            const int i = (p + k - half) / up;
            acc += (double)h[k] * (double)x[i];
        }
        out[m] = (float)acc;
    }
    return out;
}

// ===========================================================================
// FFT (DFT-N for arbitrary N=960 — n_fft=960 is not a power of two, so we
// fall back to the same recursive split that voxtral uses, with a base
// case of plain DFT for non-power-of-two leaves). Cooley-Tukey on N=960
// = 64 * 15 splits cleanly to power-of-two on the even side and DFT on
// the odd (15) side.
// ===========================================================================

static void mimo_dft_naive(const float* in, int N, float* out) {
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

static void mimo_fft_inplace_split(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half = N / 2;
    if (N - half * 2 == 1) {
        mimo_dft_naive(in, N, out);
        return;
    }
    // even/odd split — borrowed from voxtral.cpp
    float* even = in + N;
    for (int i = 0; i < half; i++)
        even[i] = in[2 * i];
    float* ef = out + 2 * N;
    mimo_fft_inplace_split(even, half, ef);
    float* odd = even;
    for (int i = 0; i < half; i++)
        odd[i] = in[2 * i + 1];
    float* of = ef + N;
    mimo_fft_inplace_split(odd, half, of);
    for (int k = 0; k < half; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float re = std::cos(ang), im = std::sin(ang);
        float reo = of[2 * k], imo = of[2 * k + 1];
        out[2 * k] = ef[2 * k] + re * reo - im * imo;
        out[2 * k + 1] = ef[2 * k + 1] + re * imo + im * reo;
        out[2 * (k + half)] = ef[2 * k] - re * reo + im * imo;
        out[2 * (k + half) + 1] = ef[2 * k + 1] - re * imo - im * reo;
    }
}

static void mimo_fft_wrapper(const float* in, int N, float* out) {
    // Match voxtral_fft_wrapper sizing: scratch_in needs 4*N, scratch_out 8*N.
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    mimo_fft_inplace_split(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

// ===========================================================================
// Compute log-mel for the tokenizer (24 kHz, magnitude, log-clip 1e-7).
// Returns (T_mel, n_mels) row-major in `out_mel` and writes T_mel to *T_mel_out.
// ===========================================================================

static std::vector<float> compute_tokenizer_mel(const float* pcm16k, int n_in, const mimo_tok_hp& hp, int& T_mel_out) {
    // 1. Resample to 24 kHz.
    std::vector<float> pcm24k = resample_sinc_hann(pcm16k, n_in, 16000, (int)hp.sampling_rate);

    // 2. Build STFT window (Hann periodic) and mel filterbank (HTK scale,
    // shared helper in core_mel).
    const int n_fft = (int)hp.nfft;
    const int win_length = (int)hp.window_size;
    const int hop = (int)hp.hop_length;
    const int n_mels = (int)hp.n_mels;
    const int n_freqs = n_fft / 2 + 1;
    std::vector<float> hann(win_length);
    for (int i = 0; i < win_length; i++)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)win_length));
    std::vector<float> mel_fb = core_mel::build_htk_fb((int)hp.sampling_rate, n_fft, n_mels, /*fmin*/ 0.0f,
                                                       /*fmax*/ -1.0f, core_mel::FbLayout::MelsFreqs);

    // 3. core_mel::compute with magnitude + log-clip 1e-7 + no normalization.
    // Output in MelsTime layout (n_mels, T) row-major — that matches ggml's
    // ne=(T, n_mels) interpretation when fed into conv1 (ne[0]=T varies
    // fastest in memory, so data[m*T+t] = tensor[t, m]). The diff-harness
    // tok_mel stage does a final transpose to TimeMels to match the
    // (T, n_mels) layout the Python ref dumper writes.
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win_length;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.log_guard = core_mel::LogGuard::MaxClip;   // log(max(spec, 1e-7))
    p.spec_kind = core_mel::SpecKind::Magnitude; // power=1.0
    p.norm = core_mel::Normalization::None;
    p.layout = core_mel::Layout::MelsTime; // (n_mels, T)
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Float;
    p.log_eps = 1e-7f;
    p.center_pad = true; // zero-pad; torchaudio uses reflect — minor mismatch
    p.drop_last_frame = false;

    int T = 0;
    std::vector<float> mel = core_mel::compute(pcm24k.data(), (int)pcm24k.size(), hann.data(), win_length,
                                               mel_fb.data(), n_freqs, mimo_fft_wrapper, p, T);
    T_mel_out = T;
    return mel;
}

} // namespace

// ===========================================================================
// API: default params + init/free.
// ===========================================================================

extern "C" struct mimo_tokenizer_context_params mimo_tokenizer_context_default_params(void) {
    mimo_tokenizer_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" struct mimo_tokenizer_context* mimo_tokenizer_init_from_file(const char* path_model,
                                                                        struct mimo_tokenizer_context_params params) {
    auto* ctx = new mimo_tokenizer_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // ---- Pass 1: metadata + hparams ----
    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.d_model = core_gguf::kv_u32(gctx, "mimo_tok.d_model", hp.d_model);
    hp.encoder_layers = core_gguf::kv_u32(gctx, "mimo_tok.encoder_layers", hp.encoder_layers);
    hp.encoder_heads = core_gguf::kv_u32(gctx, "mimo_tok.encoder_heads", hp.encoder_heads);
    hp.encoder_ffn_dim = core_gguf::kv_u32(gctx, "mimo_tok.encoder_ffn_dim", hp.encoder_ffn_dim);
    hp.num_quantizers = core_gguf::kv_u32(gctx, "mimo_tok.num_quantizers", hp.num_quantizers);
    hp.sampling_rate = core_gguf::kv_u32(gctx, "mimo_tok.sampling_rate", hp.sampling_rate);
    hp.hop_length = core_gguf::kv_u32(gctx, "mimo_tok.hop_length", hp.hop_length);
    hp.stride_size = core_gguf::kv_u32(gctx, "mimo_tok.stride_size", hp.stride_size);
    hp.avg_pooler = core_gguf::kv_u32(gctx, "mimo_tok.avg_pooler", hp.avg_pooler);
    hp.kernel_size = core_gguf::kv_u32(gctx, "mimo_tok.kernel_size", hp.kernel_size);
    // Pull codebook sizes (the converter writes them per index).
    hp.codebook_size.clear();
    for (uint32_t i = 0; i < hp.num_quantizers; i++) {
        char key[64];
        snprintf(key, sizeof(key), "mimo_tok.codebook_size.%u", i);
        hp.codebook_size.push_back(core_gguf::kv_u32(gctx, key, 0));
    }
    core_gguf::free_metadata(gctx);

    // ---- Backends ----
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "mimo_tokenizer: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    // ---- Pass 2: weights (load to CPU; encoder forward graph picks per-op) ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend_cpu, "mimo_tokenizer", wl)) {
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    // ---- Bind named tensors. Conv stem and final norm use the upstream
    // `encoder.*` prefix (the converter's rename rules don't touch them
    // because the upstream uses `conv1`/`conv2`/`down_sample_layer`, which
    // the rename list doesn't match). Per-layer tensors land at `enc.blk.*`.
    auto& T = ctx->tensors;
    auto bind = [&](const char* name) -> ggml_tensor* { return core_gguf::require(T, name, "mimo_tokenizer"); };

    ctx->conv1_w = bind("encoder.conv1.weight");
    ctx->conv1_b = bind("encoder.conv1.bias");
    ctx->conv2_w = bind("encoder.conv2.weight");
    ctx->conv2_b = bind("encoder.conv2.bias");
    ctx->final_norm_w = bind("enc.norm.weight");
    ctx->final_norm_b = bind("enc.norm.bias");
    ctx->down_w = bind("encoder.down_sample_layer.0.weight");
    ctx->down_norm_w = bind("encoder.down_sample_norm.weight");
    ctx->down_norm_b = bind("encoder.down_sample_norm.bias");

    ctx->layers.resize(hp.encoder_layers);
    char buf[128];
    for (uint32_t i = 0; i < hp.encoder_layers; i++) {
        auto& L = ctx->layers[i];
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn_norm.weight", i);
        L.attn_norm_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn_norm.bias", i);
        L.attn_norm_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.q.weight", i);
        L.q_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.q.bias", i);
        L.q_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.k.weight", i);
        L.k_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.v.weight", i);
        L.v_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.v.bias", i);
        L.v_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.o.weight", i);
        L.o_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.o.bias", i);
        L.o_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.ffn_norm.weight", i);
        L.ffn_norm_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.ffn_norm.bias", i);
        L.ffn_norm_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.fc1.weight", i);
        L.fc1_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.fc1.bias", i);
        L.fc1_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.fc2.weight", i);
        L.fc2_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.fc2.bias", i);
        L.fc2_b = bind(buf);
    }

    // RVQ codebooks. Only the first 8 are needed for ASR; bind any present.
    ctx->codebooks.resize(hp.num_quantizers);
    for (uint32_t i = 0; i < hp.num_quantizers; i++) {
        snprintf(buf, sizeof(buf), "encoder.quant.vq.layers.%u._codebook.embed", i);
        ggml_tensor* e = core_gguf::try_get(ctx->tensors, buf);
        if (e) {
            ctx->codebooks[i].embed = e;
            // GGUF prints shape [d_model, codebook_size] (innermost first);
            // ne[0] = d_model, ne[1] = codebook_size.
            ctx->codebooks[i].codebook_size = (uint32_t)e->ne[1];
        }
    }

    // ---- Backend scheduler (multi-backend; per-op placement decided at
    // graph build time). 16k nodes covers the 32-layer encoder with room
    // to spare.
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, /*graph_size=*/16384,
                                            /*parallel=*/false, /*op_offload=*/false);
        if (!ctx->sched) {
            fprintf(stderr, "mimo_tokenizer: failed to allocate scheduler\n");
            delete ctx;
            return nullptr;
        }
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "mimo_tokenizer: loaded %zu tensors  encoder=%uL/%u  rvq=%u stages\n", ctx->tensors.size(),
                hp.encoder_layers, hp.d_model, hp.num_quantizers);
    }
    return ctx;
}

extern "C" void mimo_tokenizer_free(struct mimo_tokenizer_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void mimo_tokenizer_set_n_threads(struct mimo_tokenizer_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0)
        return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}

// ===========================================================================
// Encoder graph.
//
// Inputs (by graph name):
//   "mel"        F32  ne=(T_mel, n_mels)             — already log-clipped
//   "positions"  I32  ne=(T_xfmr,)                   — RoPE position ids
//
// Stage outputs (by graph name):
//   "tok_conv1_out"  F32  ne=(T_mel,    d_model, 1)
//   "tok_conv2_out"  F32  ne=(T_xfmr,   d_model, 1)
//   "tok_xfmr_out"   F32  ne=(d_model,  T_xfmr)
//   "tok_pool_out"   F32  ne=(d_model,  T_pool)      — final encoder output
// ===========================================================================

static ggml_cgraph* mimo_tok_build_encoder_graph(mimo_tokenizer_context* ctx, int T_mel, int T_xfmr, int T_pool) {
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;               // 1280
    const int n_q = (int)hp.encoder_heads;       // 20
    const int hd = d / n_q;                      // 64
    const int n_layers = (int)hp.encoder_layers; // 32
    const int n_mels = (int)hp.n_mels;
    const int skip_after = (int)hp.encoder_skip_layer_id - 1; // = 2 (0-indexed)
    const int avg_pooler = (int)hp.avg_pooler;                // 2
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ---- Inputs ----
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_xfmr);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ---- Conv stem (mirrors voxtral) ----
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    // conv1: (128 → 1280, k=3, p=1). Output ne = (T_mel, 1280, 1).
    ggml_tensor* cur = ggml_conv_1d(ctx0, ctx->conv1_w, mel, /*s*/ 1, /*p*/ 1, /*d*/ 1);
    cur = ggml_add(ctx0, cur, bias_1d(ctx->conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);
    ggml_set_name(cur, "tok_conv1_out");
    ggml_set_output(cur); // expose via graph

    // conv2: (1280 → 1280, k=3, s=2, p=1). Output ne = (T_xfmr, 1280, 1).
    cur = ggml_conv_1d(ctx0, ctx->conv2_w, cur, /*s*/ 2, /*p*/ 1, /*d*/ 1);
    cur = ggml_add(ctx0, cur, bias_1d(ctx->conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);
    ggml_set_name(cur, "tok_conv2_out");
    ggml_set_output(cur);

    // Reshape (T_xfmr, 1280, 1) → (T_xfmr, 1280) → transpose to (1280, T_xfmr).
    cur = ggml_reshape_2d(ctx0, cur, T_xfmr, d);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    // ---- 32-layer pre-LN transformer with skip-add ----
    ggml_tensor* skip = nullptr;
    for (int il = 0; il < n_layers; il++) {
        const auto& L = ctx->layers[il];
        ggml_tensor* residual = cur;

        // Pre-attention LN
        ggml_tensor* x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, L.attn_norm_w);
        x = ggml_add(ctx0, x, L.attn_norm_b);

        // Self-attention with biased Q/V/O, no K bias, RoPE θ=10000.
        core_attn::EncoderSelfAttnParams ap;
        ap.n_heads = n_q;
        ap.n_kv_heads = n_q; // MHA
        ap.head_dim = hd;
        ap.n_kv_grp = 1;
        ap.attn_scale = attn_scale;
        ap.n_ctx_orig = 0; // unused with no scaling
        ap.rope_theta = hp.rope_theta;
        ggml_tensor* attn = core_attn::encoder_self_attn(ctx0, x, L.q_w, L.q_b, L.k_w, /*k_b*/ nullptr, L.v_w, L.v_b,
                                                         L.o_w, L.o_b, positions, /*mask*/ nullptr, ap);
        cur = ggml_add(ctx0, residual, attn);

        // Pre-FFN LN + plain GELU MLP with biases (NOT SwiGLU).
        residual = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, L.ffn_norm_w);
        x = ggml_add(ctx0, x, L.ffn_norm_b);
        x = core_ffn::gelu_erf_ffn(ctx0, x, L.fc1_w, L.fc1_b, L.fc2_w, L.fc2_b);
        cur = ggml_add(ctx0, residual, x);

        if (il == skip_after) {
            // Save a clone after layer 2 for the post-stack skip-add.
            skip = ggml_dup(ctx0, cur);
        }
    }

    // skip-add then final LayerNorm.
    if (skip) {
        cur = ggml_add(ctx0, cur, skip);
    }
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, ctx->final_norm_w);
    cur = ggml_add(ctx0, cur, ctx->final_norm_b);
    // cur ne = (d, T_xfmr)
    ggml_set_name(cur, "tok_xfmr_out");
    ggml_set_output(cur);

    // ---- down_sample: pad-to-multiple → Conv1d(k=2,s=2,no bias) → GELU → LayerNorm
    // Pad to T_xfmr_padded by appending zeros on the right (no-op when even).
    const int T_xfmr_padded = (T_xfmr + avg_pooler - 1) / avg_pooler * avg_pooler;
    if (T_xfmr_padded != T_xfmr) {
        // cur ne = (d, T_xfmr); the time axis is ne[1], so pad p1.
        cur = ggml_pad(ctx0, cur, 0, T_xfmr_padded - T_xfmr, 0, 0);
    }
    // Conv1d expects (T, C_in) layout. cur is (d, T_padded); transpose.
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (T_padded, d)
    cur = ggml_conv_1d(ctx0, ctx->down_w, cur, /*s*/ 2, /*p*/ 0, /*d*/ 1);
    // No bias.
    cur = ggml_gelu_erf(ctx0, cur);
    // cur ne = (T_pool, d, 1) → squeeze + transpose to (d, T_pool) for LN.
    cur = ggml_reshape_2d(ctx0, cur, T_pool, d);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    // LayerNorm
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, ctx->down_norm_w);
    cur = ggml_add(ctx0, cur, ctx->down_norm_b);
    ggml_set_name(cur, "tok_pool_out");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Run encoder graph and pull stage outputs back to host. Returns true on
// success; populates the output vectors with row-major (T, D) F32 data.
// ===========================================================================

namespace {

struct EncoderOutputs {
    int T_mel = 0;
    int T_xfmr = 0;
    int T_pool = 0;
    std::vector<float> conv1_out; // (T_mel, d)
    std::vector<float> conv2_out; // (T_xfmr, d)
    std::vector<float> xfmr_out;  // (T_xfmr, d)
    std::vector<float> pool_out;  // (T_pool, d)
};

static bool run_encoder(mimo_tokenizer_context* ctx, const float* pcm16k, int n_in, EncoderOutputs& out,
                        bool need_conv1, bool need_conv2, bool need_xfmr, bool need_pool) {
    if (!ctx || !pcm16k || n_in <= 0)
        return false;
    const auto& hp = ctx->hp;

    // 1. CPU-side mel.
    int T_mel = 0;
    std::vector<float> mel = compute_tokenizer_mel(pcm16k, n_in, hp, T_mel);
    if (T_mel <= 0) {
        fprintf(stderr, "mimo_tokenizer: input audio too short for STFT\n");
        return false;
    }
    out.T_mel = T_mel;
    // Compute T_xfmr and T_pool from the conv geometry (must match graph).
    // conv2: stride=2, pad=1, k=3 → T_out = floor((T_mel + 2 - 3)/2 + 1) = floor((T_mel - 1)/2 + 1).
    const int T_xfmr = (T_mel - 1) / 2 + 1;
    const int avg = (int)hp.avg_pooler;
    const int T_xfmr_padded = (T_xfmr + avg - 1) / avg * avg;
    const int T_pool = T_xfmr_padded / avg;
    out.T_xfmr = T_xfmr;
    out.T_pool = T_pool;

    // 2. Build graph.
    ggml_cgraph* gf = mimo_tok_build_encoder_graph(ctx, T_mel, T_xfmr, T_pool);

    // 3. Schedule + alloc.
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "mimo_tokenizer: failed to alloc encoder graph\n");
        return false;
    }

    // 4. Set inputs.
    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> positions(T_xfmr);
    std::iota(positions.begin(), positions.end(), 0);
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));

    // 5. Compute.
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "mimo_tokenizer: encoder graph compute failed\n");
        return false;
    }

    // 6. Extract requested stages, normalising to TimeMels (T, D) row-major
    // to match the Python ref dumper's `permute(0, 2, 1)` / `.T` outputs.
    //
    //   ggml conv output: ne=(T, D, 1) → memory data[d*T + t] (MelsTime)
    //                     → transpose to TimeMels (T, D) for the dump.
    //   ggml ne=(D, T) post-transpose: memory data[t*D + d] = TimeMels already.
    auto pull = [&](const char* name, std::vector<float>& dst, int T_dim, int D_dim, bool need_transpose) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t) {
            fprintf(stderr, "mimo_tokenizer: missing graph output '%s'\n", name);
            return false;
        }
        const size_t total = ggml_nelements(t) * sizeof(float);
        std::vector<float> raw(ggml_nelements(t));
        ggml_backend_tensor_get(t, raw.data(), 0, total);
        if (need_transpose) {
            // raw is MelsTime [d * T + t]; dst should be TimeMels [t * D + d].
            dst.assign((size_t)T_dim * D_dim, 0.0f);
            for (int d = 0; d < D_dim; d++)
                for (int tt = 0; tt < T_dim; tt++)
                    dst[(size_t)tt * D_dim + d] = raw[(size_t)d * T_dim + tt];
        } else {
            dst = std::move(raw);
        }
        return true;
    };

    const int d = (int)hp.d_model;
    if (need_conv1 && !pull("tok_conv1_out", out.conv1_out, T_mel, d, /*transpose*/ true))
        return false;
    if (need_conv2 && !pull("tok_conv2_out", out.conv2_out, T_xfmr, d, /*transpose*/ true))
        return false;
    if (need_xfmr && !pull("tok_xfmr_out", out.xfmr_out, T_xfmr, d, /*transpose*/ false))
        return false;
    if (need_pool && !pull("tok_pool_out", out.pool_out, T_pool, d, /*transpose*/ false))
        return false;

    return true;
}

// ===========================================================================
// 8-stage Euclidean RVQ encode. CPU-side argmin over ||x - embed[k]||^2.
// Codebooks are F16 on disk; promoted to F32 on first use (mirrors the
// upstream `quantizer.float()` call).
// ===========================================================================

static void ensure_codebook_f32(mimo_tokenizer_context* ctx, int stage_idx) {
    auto& cb = ctx->codebooks[stage_idx];
    if (!cb.embed_f32.empty() || !cb.embed)
        return;
    const int d = (int)ctx->hp.d_model;
    const int K = (int)cb.codebook_size;
    cb.embed_f32.assign((size_t)K * d, 0.0f);
    // GGUF on-disk layout: ne[0]=d_model, ne[1]=codebook_size, F16.
    // We want row-major (K, d) for fast argmin.
    if (cb.embed->type == GGML_TYPE_F16) {
        std::vector<uint16_t> raw((size_t)K * d);
        ggml_backend_tensor_get(cb.embed, raw.data(), 0, raw.size() * sizeof(uint16_t));
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < d; j++) {
                cb.embed_f32[(size_t)k * d + j] = ggml_fp16_to_fp32((ggml_fp16_t)raw[(size_t)k * d + j]);
            }
        }
    } else if (cb.embed->type == GGML_TYPE_F32) {
        std::vector<float> raw((size_t)K * d);
        ggml_backend_tensor_get(cb.embed, raw.data(), 0, raw.size() * sizeof(float));
        cb.embed_f32 = std::move(raw); // already (K, d)
    } else {
        fprintf(stderr, "mimo_tokenizer: unsupported codebook dtype %d for stage %d\n", (int)cb.embed->type, stage_idx);
        return;
    }
    cb.embed_norm_sq.assign(K, 0.0f);
    for (int k = 0; k < K; k++) {
        float s = 0.0f;
        for (int j = 0; j < d; j++) {
            float v = cb.embed_f32[(size_t)k * d + j];
            s += v * v;
        }
        cb.embed_norm_sq[k] = s;
    }
}

static std::vector<int32_t> rvq_encode(mimo_tokenizer_context* ctx, const float* features, int T) {
    const int d = (int)ctx->hp.d_model;
    const int n_stages = 8;
    if ((int)ctx->codebooks.size() < n_stages) {
        fprintf(stderr, "mimo_tokenizer: only %zu codebooks bound, need %d\n", ctx->codebooks.size(), n_stages);
        return {};
    }
    std::vector<core_rvq::Codebook> stages(n_stages);
    for (int s = 0; s < n_stages; s++) {
        if (!ctx->codebooks[s].embed) {
            fprintf(stderr, "mimo_tokenizer: missing codebook for stage %d\n", s);
            return {};
        }
        ensure_codebook_f32(ctx, s);
        const auto& cb = ctx->codebooks[s];
        stages[s].embed = cb.embed_f32.data();
        stages[s].embed_norm_sq = cb.embed_norm_sq.data();
        stages[s].codebook_size = (int)cb.codebook_size;
        stages[s].dim = d;
    }

    std::vector<int32_t> codes((size_t)T * n_stages, 0);
    if (!core_rvq::encode_euclidean(features, T, d, stages.data(), n_stages, codes.data()))
        return {};
    return codes;
}

} // namespace

// ===========================================================================
// Public extract_stage / encode_pcm16k.
// ===========================================================================

extern "C" int32_t* mimo_tokenizer_encode_pcm16k(struct mimo_tokenizer_context* ctx, const float* pcm, int n_samples,
                                                 int* n_frames_out) {
    if (n_frames_out)
        *n_frames_out = 0;
    if (!ctx)
        return nullptr;
    EncoderOutputs eo;
    if (!run_encoder(ctx, pcm, n_samples, eo, /*conv1*/ false, /*conv2*/ false, /*xfmr*/ false, /*pool*/ true))
        return nullptr;
    auto codes = rvq_encode(ctx, eo.pool_out.data(), eo.T_pool);
    if (codes.empty())
        return nullptr;
    int32_t* result = (int32_t*)std::malloc(codes.size() * sizeof(int32_t));
    std::memcpy(result, codes.data(), codes.size() * sizeof(int32_t));
    if (n_frames_out)
        *n_frames_out = eo.T_pool;
    return result;
}

extern "C" float* mimo_tokenizer_extract_stage(struct mimo_tokenizer_context* ctx, const float* pcm, int n_samples,
                                               const char* stage, int* n_out) {
    if (n_out)
        *n_out = 0;
    if (!ctx || !stage)
        return nullptr;
    const std::string s = stage;

    auto give = [&](const std::vector<float>& v) -> float* {
        float* r = (float*)std::malloc(v.size() * sizeof(float));
        std::memcpy(r, v.data(), v.size() * sizeof(float));
        if (n_out)
            *n_out = (int)v.size();
        return r;
    };

    if (s == "tok_mel") {
        int T_mel = 0;
        std::vector<float> mel_mt = compute_tokenizer_mel(pcm, n_samples, ctx->hp, T_mel);
        if (T_mel <= 0)
            return nullptr;
        // Transpose MelsTime (n_mels, T) → TimeMels (T, n_mels) to match the
        // Python ref dumper's `log_mel.squeeze(0).T.contiguous()` layout.
        const int n_mels = (int)ctx->hp.n_mels;
        std::vector<float> mel_tm((size_t)T_mel * n_mels);
        for (int m = 0; m < n_mels; m++)
            for (int t = 0; t < T_mel; t++)
                mel_tm[(size_t)t * n_mels + m] = mel_mt[(size_t)m * T_mel + t];
        return give(mel_tm);
    }

    EncoderOutputs eo;
    const bool want_conv1 = (s == "tok_conv1_out");
    const bool want_conv2 = (s == "tok_conv2_out");
    const bool want_xfmr = (s == "tok_xfmr_out");
    const bool want_pool = (s == "tok_pool_out") || (s == "tok_codes");
    if (!want_conv1 && !want_conv2 && !want_xfmr && !want_pool) {
        fprintf(stderr, "mimo_tokenizer_extract_stage: unknown stage '%s'\n", stage);
        return nullptr;
    }
    if (!run_encoder(ctx, pcm, n_samples, eo, want_conv1, want_conv2, want_xfmr, want_pool))
        return nullptr;
    if (want_conv1)
        return give(eo.conv1_out);
    if (want_conv2)
        return give(eo.conv2_out);
    if (want_xfmr)
        return give(eo.xfmr_out);
    if (s == "tok_pool_out")
        return give(eo.pool_out);
    if (s == "tok_codes") {
        auto codes = rvq_encode(ctx, eo.pool_out.data(), eo.T_pool);
        if (codes.empty())
            return nullptr;
        // Cast to float32 row-major (T_pool, 8) for cosine-comparable output.
        std::vector<float> codes_f((size_t)eo.T_pool * 8);
        for (size_t i = 0; i < codes_f.size(); i++)
            codes_f[i] = (float)codes[i];
        return give(codes_f);
    }
    return nullptr;
}
