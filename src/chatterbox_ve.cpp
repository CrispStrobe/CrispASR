// chatterbox_ve.cpp — VoiceEncoder (VE) forward pass for native voice cloning.
//
// Module 2 of the chatterbox WAV→cond port. Mirrors upstream
// `chatterbox.models.voice_encoder.embeds_from_wavs` minus the silence
// trim (deferred). See chatterbox_ve.h for the pipeline contract.

#include "chatterbox_ve.h"

#include "core/fft.h"
#include "core/mel.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox_ve {

namespace {

// VE config constants, matching `voice_encoder/config.py`.
constexpr int kSampleRate = 16000;
constexpr int kNFft = 400;
constexpr int kHop = 160;
constexpr int kWin = 400;
constexpr int kNMels = 40;
constexpr float kFmin = 0.0f;
constexpr float kFmax = 8000.0f;
constexpr int kPartialFrames = 160;
constexpr int kHidden = 256;
constexpr int kEmbed = 256;
constexpr float kDefaultRate = 1.3f; // Resemble's `embeds_from_wavs` default

// Periodic Hann window (matches `librosa.stft(window='hann')` /
// `scipy.signal.windows.get_window('hann', N, fftbins=True)`):
//   w[i] = 0.5 * (1 - cos(2π i / N))   for i ∈ [0, N)
static void make_hann_periodic(int N, std::vector<float>& out) {
    out.resize((size_t)N);
    for (int i = 0; i < N; i++) {
        out[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)N));
    }
}

// Compute partial-window stride for `embeds_from_wavs` (rate-driven,
// not overlap-driven). Matches `voice_encoder.get_frame_step(rate=1.3)`:
//   frame_step = round((sample_rate / rate) / partial_frames)
//             = round((16000 / 1.3) / 160) = 77
static int compute_frame_step(float rate) {
    const float raw = ((float)kSampleRate / rate) / (float)kPartialFrames;
    return (int)std::lround(raw);
}

// Decide how many overlapping 160-frame partials fit in `T` mel frames.
// Mirrors `voice_encoder.get_num_wins(min_coverage=0.8)`:
//   n_wins, remainder = divmod(max(T - 160 + step, 0), step)
//   if first window short enough, add one more if coverage ≥ 0.8 of a partial
static void plan_partials(int T, int step, int& n_partials, int& target_T) {
    if (T <= 0) {
        n_partials = 0;
        target_T = 0;
        return;
    }
    const int win = kPartialFrames;
    const float min_cov = 0.8f;
    const int x = std::max(T - win + step, 0);
    int n = x / step;
    const int rem = x % step;
    if (n == 0 || ((float)(rem + (win - step)) / (float)win) >= min_cov) {
        n += 1;
    }
    n_partials = n;
    target_T = win + step * (n - 1);
}

// Read any tensor (F32 / F16 / quantized) back from its backend buffer into a
// freshly-sized float32 vector. Mirrors the `tensor_get_f32` helper in
// chatterbox.cpp; duplicated here so chatterbox_ve.cpp stays self-contained
// and we don't expose that helper across the anonymous namespace boundary.
static std::vector<float> read_tensor_f32(ggml_tensor* t) {
    const size_t n = (size_t)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return out;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t)n);
        return out;
    }
    // Quantized — read raw bytes then dequantize via ggml CPU traits.
    const size_t raw = ggml_nbytes(t);
    std::vector<uint8_t> tmp(raw);
    ggml_backend_tensor_get(t, tmp.data(), 0, raw);
    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (tt && tt->to_float) {
        tt->to_float(tmp.data(), out.data(), (int64_t)n);
    } else {
        std::fill(out.begin(), out.end(), 0.0f);
    }
    return out;
}

// Pure-CPU sigmoid / tanh that match torch's default fp32 behaviour.
static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
static inline float tanhf_local(float x) {
    return std::tanh(x);
}

// Single-direction LSTM forward (matches PyTorch nn.LSTM with batch_first=True
// and num_layers=1). Storage convention mirrors the GGUF layout used by every
// LSTM in this codebase:
//
//   W_ih: PyTorch (4H, in_dim) row-major  → ggml ne=(in_dim, 4H)
//   W_hh: PyTorch (4H, H)      row-major  → ggml ne=(H, 4H)
//   b_ih: (4H,) F32
//   b_hh: (4H,) F32
//
// Reads are flat row-major as PyTorch saved them. Writes outputs as
// (T, H) row-major. Gate order is PyTorch's (i, f, g, o).
static void lstm_unidir_cpu(const float* X_in, int T, int in_dim, int H, const float* W_ih, const float* W_hh,
                            const float* b_ih, const float* b_hh, float* out_th) {
    const int H4 = 4 * H;
    // Pre-compute proj_x = X @ W_ih^T + b_ih over all T at once → (T, 4H).
    // PyTorch: pre_t = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh.
    // W_ih is (4H, in_dim); we apply via inner-loop dot-product.
    std::vector<float> proj(T * H4);
    for (int t = 0; t < T; t++) {
        const float* x = X_in + (size_t)t * (size_t)in_dim;
        float* p = proj.data() + (size_t)t * (size_t)H4;
        for (int j = 0; j < H4; j++) {
            const float* w = W_ih + (size_t)j * (size_t)in_dim; // row j
            float s = b_ih[j];
            for (int k = 0; k < in_dim; k++)
                s += w[k] * x[k];
            p[j] = s;
        }
    }

    std::vector<float> h(H, 0.0f);
    std::vector<float> c(H, 0.0f);
    std::vector<float> pre(H4);
    for (int t = 0; t < T; t++) {
        const float* p = proj.data() + (size_t)t * (size_t)H4;
        // pre = proj_x[t] + (W_hh @ h + b_hh)
        for (int j = 0; j < H4; j++) {
            const float* w = W_hh + (size_t)j * (size_t)H; // row j
            float s = b_hh[j];
            for (int k = 0; k < H; k++)
                s += w[k] * h[k];
            pre[j] = p[j] + s;
        }
        const float* gi = pre.data() + 0 * H;
        const float* gf = pre.data() + 1 * H;
        const float* gg = pre.data() + 2 * H;
        const float* go = pre.data() + 3 * H;
        for (int k = 0; k < H; k++) {
            const float i_t = sigmoidf(gi[k]);
            const float f_t = sigmoidf(gf[k]);
            const float g_t = tanhf_local(gg[k]);
            const float o_t = sigmoidf(go[k]);
            c[k] = f_t * c[k] + i_t * g_t;
            h[k] = o_t * tanhf_local(c[k]);
        }
        float* outr = out_th + (size_t)t * (size_t)H;
        std::memcpy(outr, h.data(), (size_t)H * sizeof(float));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Mel — plain CPU compute via core_mel + core_fft.
// ---------------------------------------------------------------------------

std::vector<float> compute_mel(const float* pcm_16k, int n_samples, int& T_out) {
    T_out = 0;
    if (!pcm_16k || n_samples <= 0)
        return {};

    static thread_local std::vector<float> hann;
    if ((int)hann.size() != kWin) {
        make_hann_periodic(kWin, hann);
    }

    // Slaney-mel filterbank (40 bins, fmin=0, fmax=8 kHz). Layout MelsFreqs
    // (n_mels rows × n_freqs columns) — what core_mel expects by default.
    static thread_local std::vector<float> mel_fb;
    if (mel_fb.empty()) {
        mel_fb = core_mel::build_slaney_fb(kSampleRate, kNFft, kNMels, kFmin, kFmax, core_mel::FbLayout::MelsFreqs);
    }

    core_mel::Params p;
    p.n_fft = kNFft;
    p.hop_length = kHop;
    p.win_length = kWin;
    p.n_mels = kNMels;
    p.spec_kind = core_mel::SpecKind::Power;      // mel_power=2 → |X|^2
    p.log_base = core_mel::LogBase::None;         // mel_type='amp' → no log
    p.norm = core_mel::Normalization::None;       // normalized_mels=False
    p.layout = core_mel::Layout::TimeMels;        // VE wants (T, 40)
    p.fb_layout = core_mel::FbLayout::MelsFreqs;  // matches build_slaney_fb default
    p.matmul = core_mel::MatmulPrecision::Double; // librosa uses float64 mel projection
    p.center_pad = true;                          // librosa center=True default
    p.preemph = 0.0f;                             // hp.preemphasis = 0
    p.log_eps = 0.0f;                             // unused with LogBase::None

    int T = 0;
    auto mel = core_mel::compute(pcm_16k, n_samples, hann.data(), kWin, mel_fb.data(), kNFft / 2 + 1,
                                 &core_fft::fft_radix2_wrapper, p, T);
    T_out = T;
    return mel;
}

// ---------------------------------------------------------------------------
// Per-partial LSTM forward.
// ---------------------------------------------------------------------------

std::vector<float> compute_partial_embeds(const cb_ve_model& ve, ggml_backend_sched_t sched,
                                          std::vector<uint8_t>& compute_meta, const float* mel_tm, int T,
                                          int& n_partials_out) {
    (void)sched;
    (void)compute_meta; // VE LSTM runs purely on CPU — see file header.
    n_partials_out = 0;
    if (!mel_tm || T <= 0) {
        return {};
    }
    if (!ve.lstm_ih_w[0] || !ve.lstm_hh_w[0] || !ve.lstm_ih_b[0] || !ve.lstm_hh_b[0] || !ve.proj_w || !ve.proj_b) {
        fprintf(stderr, "chatterbox_ve: VE tensors missing — bind_ve was not called or weights absent\n");
        return {};
    }

    const int step = compute_frame_step(kDefaultRate);
    int n_partials = 0;
    int target_T = 0;
    plan_partials(T, step, n_partials, target_T);
    if (n_partials <= 0) {
        return {};
    }

    // If the mel is shorter than target_T frames, zero-pad to match
    // `voice_encoder.stride_as_partials`. If longer, the per-partial loop
    // implicitly drops the trailing remainder.
    std::vector<float> mel_padded;
    const float* mel_use = mel_tm;
    if (target_T > T) {
        mel_padded.assign((size_t)target_T * (size_t)kNMels, 0.0f);
        std::memcpy(mel_padded.data(), mel_tm, (size_t)T * kNMels * sizeof(float));
        mel_use = mel_padded.data();
    }

    // Read all VE weights to CPU float32 once (the LSTM math runs on CPU
    // regardless of where the weights live). Cheap — the full VE weight
    // bundle is ~2 MB.
    std::vector<float> W_ih[3], W_hh[3], b_ih[3], b_hh[3];
    for (int l = 0; l < 3; l++) {
        W_ih[l] = read_tensor_f32(ve.lstm_ih_w[l]);
        W_hh[l] = read_tensor_f32(ve.lstm_hh_w[l]);
        b_ih[l] = read_tensor_f32(ve.lstm_ih_b[l]);
        b_hh[l] = read_tensor_f32(ve.lstm_hh_b[l]);
    }
    std::vector<float> proj_w_f32 = read_tensor_f32(ve.proj_w);
    std::vector<float> proj_b_f32 = read_tensor_f32(ve.proj_b);

    std::vector<float> out_embeds((size_t)n_partials * (size_t)kEmbed, 0.0f);

    // Buffers shared across partials (sized to layer 0 / layer 1+ output).
    std::vector<float> layer_in((size_t)kPartialFrames * (size_t)kHidden);
    std::vector<float> layer_out((size_t)kPartialFrames * (size_t)kHidden);

    for (int p = 0; p < n_partials; p++) {
        const int t_start = p * step;
        // mel slice: (kPartialFrames, kNMels) row-major.
        const float* part_src = mel_use + (size_t)t_start * (size_t)kNMels;

        // Layer 0: in_dim=kNMels=40, out=kHidden=256.
        lstm_unidir_cpu(part_src, kPartialFrames, kNMels, kHidden, W_ih[0].data(), W_hh[0].data(), b_ih[0].data(),
                        b_hh[0].data(), layer_out.data());
        // Layers 1, 2: in_dim=out=kHidden=256, fed by previous layer's full
        // (T, H) output.
        for (int l = 1; l < 3; l++) {
            std::swap(layer_in, layer_out);
            lstm_unidir_cpu(layer_in.data(), kPartialFrames, kHidden, kHidden, W_ih[l].data(), W_hh[l].data(),
                            b_ih[l].data(), b_hh[l].data(), layer_out.data());
        }

        // Take last timestep h[T-1, :] of the final layer.
        const float* h_final = layer_out.data() + (size_t)(kPartialFrames - 1) * (size_t)kHidden;

        // Project: y = proj_w @ h_final + proj_b. proj_w is (out=256, in=256)
        // PyTorch row-major.
        float y[kEmbed];
        for (int j = 0; j < kEmbed; j++) {
            const float* w = proj_w_f32.data() + (size_t)j * (size_t)kHidden; // row j
            float s = proj_b_f32[(size_t)j];
            for (int k = 0; k < kHidden; k++) {
                s += w[k] * h_final[k];
            }
            // ve_final_relu = True
            y[j] = (s > 0.0f) ? s : 0.0f;
        }

        // L2-normalise → write into the per-partial output row.
        double sq = 0.0;
        for (int j = 0; j < kEmbed; j++) {
            sq += (double)y[j] * (double)y[j];
        }
        const float inv = (sq > 0.0) ? (float)(1.0 / std::sqrt(sq)) : 1.0f;
        float* dst = out_embeds.data() + (size_t)p * (size_t)kEmbed;
        for (int j = 0; j < kEmbed; j++) {
            dst[j] = y[j] * inv;
        }
    }

    n_partials_out = n_partials;
    return out_embeds;
}

// ---------------------------------------------------------------------------
// Full chain — PCM → 256-d L2-normed speaker embedding.
// ---------------------------------------------------------------------------

bool compute_speaker_emb(const cb_ve_model& ve, ggml_backend_sched_t sched, std::vector<uint8_t>& compute_meta,
                         const float* pcm_16k, int n_samples, float out_emb[256]) {
    if (!out_emb || !pcm_16k || n_samples <= 0)
        return false;

    int T = 0;
    auto mel = compute_mel(pcm_16k, n_samples, T);
    if (mel.empty() || T <= 0) {
        return false;
    }

    int n_partials = 0;
    auto partials = compute_partial_embeds(ve, sched, compute_meta, mel.data(), T, n_partials);
    if (partials.empty() || n_partials <= 0) {
        return false;
    }

    // Mean across partials, then L2-normalise.
    double acc[kEmbed] = {0};
    for (int p = 0; p < n_partials; p++) {
        const float* row = partials.data() + (size_t)p * (size_t)kEmbed;
        for (int i = 0; i < kEmbed; i++)
            acc[i] += (double)row[i];
    }
    const double inv_n = 1.0 / (double)n_partials;
    double sq = 0.0;
    for (int i = 0; i < kEmbed; i++) {
        acc[i] *= inv_n;
        sq += acc[i] * acc[i];
    }
    const double inv_l2 = (sq > 0.0) ? 1.0 / std::sqrt(sq) : 1.0;
    for (int i = 0; i < kEmbed; i++) {
        out_emb[i] = (float)(acc[i] * inv_l2);
    }
    return true;
}

} // namespace chatterbox_ve
