// chatterbox_s3tok.cpp — S3Tokenizer V2 forward for native voice cloning.
//
// Module 3 of the chatterbox WAV→cond port. Companion to chatterbox_ve.cpp
// (module 2); generates the speech-token streams that populate
// `conds.gen.prompt_token` (full audio) and `conds.t3.speech_prompt_tokens`
// (first 6 s, capped at 150 tokens) in the runtime conds bundle.
//
// See chatterbox_s3tok.h for the pipeline contract.

#include "chatterbox_s3tok.h"

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

namespace chatterbox_s3tok {

namespace {

constexpr int kSampleRate = 16000;
constexpr int kNFft = 400;
constexpr int kHop = 160;
constexpr int kNMels = 128;
constexpr int kHidden = 1280;
constexpr int kNHead = 20;
constexpr int kHeadDim = 64;
constexpr int kNLayer = 6;
constexpr int kFsmnK = 31;
constexpr int kFsmnPad = (kFsmnK - 1) / 2; // 15 left, 15 right (sym)
constexpr int kMlpInner = kHidden * 4;     // 5120
constexpr int kProjDownDim = 8;
constexpr int kCodebookSize = 6561; // 3^8
constexpr float kRopeTheta = 10000.0f;
constexpr int kRopeMaxPos = 2048;         // freqs_cis end (1024 * 2) — model_v2.py
constexpr float kAttnScale = 1.0f / 8.0f; // 1/sqrt(head_dim=64) = 1/8 = 0.125
constexpr float kAttnLnEps = 1e-6f;
constexpr float kMlpLnEps = 1e-5f;
constexpr int kConvStride = 2;

// Periodic Hann window (matches `torch.hann_window(n_fft)` default
// `periodic=True`): w[i] = 0.5 * (1 - cos(2π i / N)).
static void make_hann_periodic(int N, std::vector<float>& out) {
    out.resize((size_t)N);
    for (int i = 0; i < N; i++) {
        out[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)N));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// log-mel features (CPU)
// ---------------------------------------------------------------------------

std::vector<float> compute_log_mel(const float* pcm_16k, int n_samples, int& T_out) {
    T_out = 0;
    if (!pcm_16k || n_samples <= 0)
        return {};

    static thread_local std::vector<float> hann;
    if ((int)hann.size() != kNFft) {
        make_hann_periodic(kNFft, hann);
    }

    // librosa.filters.mel(sr=16000, n_fft=400, n_mels=128) — defaults: htk=False,
    // norm='slaney'. core_mel::build_slaney_fb matches.
    static thread_local std::vector<float> mel_fb;
    if (mel_fb.empty()) {
        mel_fb = core_mel::build_slaney_fb(kSampleRate, kNFft, kNMels, /*fmin*/ 0.0f, /*fmax*/ 8000.0f,
                                           core_mel::FbLayout::MelsFreqs);
    }

    // Upstream's `log_mel_spectrogram`:
    //   stft = torch.stft(audio, n_fft=400, hop=160, window=hann, return_complex=True)
    //   magnitudes = stft[..., :-1].abs()**2     # drop the last STFT frame
    //   mel = mel_filters @ magnitudes
    //   log_spec = clamp(mel, min=1e-10).log10()
    //   log_spec = max(log_spec, log_spec.max() - 8.0)
    //   log_spec = (log_spec + 4.0) / 4.0
    //
    // That matches core_mel with:
    //   spec_kind = Power
    //   log_base  = Log10
    //   log_guard = MaxClip (log_eps = 1e-10)
    //   norm      = GlobalClipMax     # → clip max-8, +4, /4
    //   layout    = MelsTime           # (n_mels, T) row-major → ggml ne=(T, n_mels)
    //   center_pad = true              # torch.stft default center=True
    //   matmul precision = Double      # match librosa's float64 mel projection
    //
    // The "drop last STFT frame" step is core_mel's `drop_last_frame` knob.
    core_mel::Params p;
    p.n_fft = kNFft;
    p.hop_length = kHop;
    p.win_length = kNFft;
    p.n_mels = kNMels;
    p.spec_kind = core_mel::SpecKind::Power;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = 1e-10f;
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.center_pad = true;
    p.preemph = 0.0f;
    p.drop_last_frame = true;

    int T = 0;
    auto mel = core_mel::compute(pcm_16k, n_samples, hann.data(), kNFft, mel_fb.data(), kNFft / 2 + 1,
                                 &core_fft::fft_radix2_wrapper, p, T);
    T_out = T;
    return mel;
}

// ---------------------------------------------------------------------------
// Encoder graph
// ---------------------------------------------------------------------------

// Build the conv front-end + 6 transformer blocks + project_down forward.
// Returns the post-project_down (T_tok, 8) tensor — caller copies it back
// to the host and runs FSQ.
//
// `mel_in_T` is the input tensor (mel) the caller has already created and
// set as input. Allocates intermediate ops onto `ctx0`/`gf`.
static ggml_tensor* build_encoder_graph(ggml_context* ctx0, ggml_cgraph* gf, const cb_s3tok_model& m,
                                        ggml_tensor* mel_in, int T) {
    // mel_in ne = (T, n_mels=128) — channel-first row-major matches the
    // chatterbox_ve / voxtral conv1d input convention. After conv1 (s=2)
    // and conv2 (s=2) we end up at ceil(ceil(T/2)/2) ≈ T/4 frames.
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    ggml_tensor* x = ggml_conv_1d(ctx0, m.conv1_w, mel_in, /*s*/ kConvStride, /*p*/ 1, /*d*/ 1);
    x = ggml_add(ctx0, x, bias_1d(m.conv1_b));
    x = ggml_gelu(ctx0, x);
    // x ne = (T/2, 1280, 1)

    x = ggml_conv_1d(ctx0, m.conv2_w, x, /*s*/ 2, /*p*/ 1, /*d*/ 1);
    x = ggml_add(ctx0, x, bias_1d(m.conv2_b));
    x = ggml_gelu(ctx0, x);
    // x ne = (T/4, 1280, 1)

    const int T_tok = (int)x->ne[0];

    // Reshape (T_tok, 1280, 1) → (1280, T_tok) for the transformer blocks
    // (everything downstream wants ne[0]=feature_dim).
    x = ggml_reshape_2d(ctx0, x, T_tok, kHidden); // (T_tok, 1280)
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (1280, T_tok)

    // Position ids for RoPE.
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_tok);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    for (int il = 0; il < kNLayer; il++) {
        const auto& b = m.blocks[il];
        ggml_tensor* residual = x;

        // attn_ln (eps=1e-6).
        ggml_tensor* h = ggml_norm(ctx0, x, kAttnLnEps);
        h = ggml_mul(ctx0, h, b.attn_ln_w);
        h = ggml_add(ctx0, h, b.attn_ln_b);

        // Q/K/V projections. S3Tokenizer follows Whisper's `MultiHeadAttention`
        // (s3tokenizer/model.py) where Q and V are biased but K has no bias —
        // the K bias would commute with softmax anyway. So `attn_k_b` is
        // expected to be nullptr for every block; guard against it.
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, h);
        if (b.attn_q_b)
            Q = ggml_add(ctx0, Q, b.attn_q_b);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, h);
        if (b.attn_k_b)
            K = ggml_add(ctx0, K, b.attn_k_b);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, h);
        if (b.attn_v_b)
            V = ggml_add(ctx0, V, b.attn_v_b);
        // Q, K, V ne = (1280, T_tok)

        // FSMN side branch on V (computed BEFORE the head reshape — it
        // operates on V flattened to (B, T, D) in the python). The
        // depthwise conv eats the T axis per-channel.
        // ggml_conv_1d_dw expects input (T, C) and weight (K, 1, C).
        // V is currently (1280, T_tok) so transpose first.
        ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, V)); // (T_tok, 1280)
        ggml_tensor* fsmn = ggml_conv_1d_dw(ctx0, b.attn_fsmn_w, v_t, /*s*/ 1, /*p*/ kFsmnPad, /*d*/ 1);
        // fsmn ne = (T_tok, 1280, 1) — squeeze the trailing singleton.
        if (ggml_n_dims(fsmn) > 2) {
            fsmn = ggml_reshape_2d(ctx0, fsmn, fsmn->ne[0], fsmn->ne[1]);
        }
        // Residual: fsmn += v_t (still in (T, D) layout). Then transpose
        // back to (D, T) so it can be added to the attention output.
        fsmn = ggml_add(ctx0, fsmn, v_t);
        fsmn = ggml_cont(ctx0, ggml_transpose(ctx0, fsmn)); // (1280, T_tok)

        // Multi-head attention with RoPE on Q/K (NEOX-style, head_dim=64).
        Q = ggml_reshape_3d(ctx0, Q, kHeadDim, kNHead, T_tok);
        K = ggml_reshape_3d(ctx0, K, kHeadDim, kNHead, T_tok);
        V = ggml_reshape_3d(ctx0, V, kHeadDim, kNHead, T_tok);

        Q = ggml_rope_ext(ctx0, Q, positions, /*freq_factors*/ nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, kRopeMaxPos,
                          kRopeTheta, /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f, /*attn_factor*/ 1.0f,
                          /*beta_fast*/ 0.0f, /*beta_slow*/ 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, kRopeMaxPos, kRopeTheta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);

        // Permute to flash-attn layout (head_dim, T, n_head, batch=1).
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, /*mask*/ nullptr, kAttnScale,
                                                /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, kHidden, T_tok); // (1280, T_tok)

        // Output projection + bias, then add FSMN memory branch.
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_out_w, attn), b.attn_out_b);
        attn = ggml_add(ctx0, attn, fsmn);

        x = ggml_add(ctx0, residual, attn);

        // mlp_ln (default eps=1e-5).
        residual = x;
        h = ggml_norm(ctx0, x, kMlpLnEps);
        h = ggml_mul(ctx0, h, b.mlp_ln_w);
        h = ggml_add(ctx0, h, b.mlp_ln_b);

        // MLP: Linear(1280→5120) + GELU + Linear(5120→1280), all biased.
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, b.mlp_up_w, h), b.mlp_up_b); // (5120, T_tok)
        h = ggml_gelu(ctx0, h);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, b.mlp_dn_w, h), b.mlp_dn_b); // (1280, T_tok)
        x = ggml_add(ctx0, x, h);
    }
    // x ne = (1280, T_tok)

    // FSQ project_down: Linear(1280, 8). quant.cb.pd.weight ggml ne=(1280, 8).
    // ggml_mul_mat(pd_w, x): x ne=(1280, T_tok), pd_w ne=(1280, 8) →
    // result ne=(8, T_tok).
    ggml_tensor* y = ggml_add(ctx0, ggml_mul_mat(ctx0, m.quant_pd_w, x), m.quant_pd_b);
    ggml_set_name(y, "s3tok_proj_down");
    ggml_build_forward_expand(gf, y);
    return y;
}

// ---------------------------------------------------------------------------
// Encoder forward — runs the graph and returns (T_tok, 8) pre-FSQ floats.
// ---------------------------------------------------------------------------

std::vector<float> encode_to_proj(const cb_s3tok_model& m, ggml_backend_sched_t sched,
                                  std::vector<uint8_t>& compute_meta, const float* mel_n_t, int T, int max_tokens,
                                  int& T_tok_out) {
    T_tok_out = 0;
    if (!sched || !mel_n_t || T <= 0)
        return {};
    if (m.blocks.size() != (size_t)kNLayer || !m.conv1_w || !m.conv2_w || !m.quant_pd_w) {
        fprintf(stderr, "chatterbox_s3tok: model not fully bound\n");
        return {};
    }

    // Truncate the mel to the max token budget (matches
    // `mel = mel[..., :max_len * 4]` upstream).
    int T_use = T;
    if (max_tokens > 0 && (max_tokens * 4) < T_use) {
        T_use = max_tokens * 4;
    }

    ggml_init_params ip = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) {
        fprintf(stderr, "chatterbox_s3tok: ggml_init failed\n");
        return {};
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: (T_use, n_mels=128) — channel-first row-major mel.
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_use, kNMels);
    ggml_set_name(mel, "s3tok_mel_in");
    ggml_set_input(mel);

    ggml_tensor* y = build_encoder_graph(ctx0, gf, m, mel, T_use);
    const int T_tok = (int)y->ne[1]; // (8, T_tok)

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "chatterbox_s3tok: failed to alloc graph\n");
        ggml_free(ctx0);
        return {};
    }
    // Set inputs.
    {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s3tok_mel_in"), mel_n_t, 0,
                                (size_t)T_use * (size_t)kNMels * sizeof(float));
        std::vector<int32_t> positions(T_tok);
        for (int i = 0; i < T_tok; i++)
            positions[i] = i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                                (size_t)T_tok * sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "chatterbox_s3tok: graph compute failed\n");
        ggml_free(ctx0);
        return {};
    }

    // Read back (T_tok, 8) — note ggml ne=(8, T_tok) so the flat byte
    // order is row-major (T_tok, 8). Each of the T_tok rows holds 8 floats.
    std::vector<float> out((size_t)T_tok * (size_t)kProjDownDim);
    ggml_tensor* yt = ggml_graph_get_tensor(gf, "s3tok_proj_down");
    ggml_backend_tensor_get(yt, out.data(), 0, out.size() * sizeof(float));
    ggml_free(ctx0);

    T_tok_out = T_tok;
    return out;
}

// ---------------------------------------------------------------------------
// FSQ codebook (CPU) and full token chain.
// ---------------------------------------------------------------------------

std::vector<int32_t> encode_tokens(const cb_s3tok_model& m, ggml_backend_sched_t sched,
                                   std::vector<uint8_t>& compute_meta, const float* mel_n_t, int T, int max_tokens) {
    int T_tok = 0;
    auto proj = encode_to_proj(m, sched, compute_meta, mel_n_t, T, max_tokens, T_tok);
    if (proj.empty() || T_tok <= 0)
        return {};

    // FSQ encode:
    //   h = tanh(proj) * 0.9990000128746033
    //   h = round(h) + 1                  # values in {0, 1, 2}
    //   token = Σ_i 3^i * h_i              # scalar in [0, 6561)
    //
    // Round-half-to-even matches torch's default tensor.round() which uses
    // the platform's banker's rounding. std::nearbyint with FE_TONEAREST
    // (the default rounding mode) does the same.
    constexpr float kFsqGain = 0.9990000128746033f;
    constexpr int kPowers[kProjDownDim] = {1, 3, 9, 27, 81, 243, 729, 2187};

    std::vector<int32_t> tokens((size_t)T_tok, 0);
    for (int t = 0; t < T_tok; t++) {
        const float* row = proj.data() + (size_t)t * (size_t)kProjDownDim;
        int code = 0;
        for (int i = 0; i < kProjDownDim; i++) {
            float h = std::tanh(row[i]) * kFsqGain;
            int v = (int)std::nearbyint(h) + 1; // {0, 1, 2}
            if (v < 0)
                v = 0;
            if (v > 2)
                v = 2;
            code += v * kPowers[i];
        }
        if (code < 0)
            code = 0;
        if (code >= kCodebookSize)
            code = kCodebookSize - 1;
        tokens[t] = code;
    }
    return tokens;
}

std::vector<int32_t> tokenize(const cb_s3tok_model& m, ggml_backend_sched_t sched, std::vector<uint8_t>& compute_meta,
                              const float* pcm_16k, int n_samples, int max_tokens) {
    int T = 0;
    auto mel = compute_log_mel(pcm_16k, n_samples, T);
    if (mel.empty() || T <= 0)
        return {};
    return encode_tokens(m, sched, compute_meta, mel.data(), T, max_tokens);
}

} // namespace chatterbox_s3tok
