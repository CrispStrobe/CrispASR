// chatterbox_s3gen.cpp — S3Gen (flow matching) + HiFTGenerator vocoder.
//
// This file implements the second and third stages of the Chatterbox pipeline:
//   Stage 2: Speech tokens → mel-spectrogram via conditional flow matching
//   Stage 3: Mel → 24 kHz waveform via HiFT-GAN vocoder
//
// Architecture (from chatterbox/models/s3gen/):
//   - UpsampleConformerEncoder: 6 pre-upsample + 4 post-upsample conformer
//     blocks with relative positional self-attention (512D, 8 heads, 2048 FFN)
//   - ConditionalDecoder: UNet1D with causal conv1d, 1 down + 12 mid + 1 up
//     blocks, each containing CausalResnetBlock1D + 4 BasicTransformerBlocks
//   - CausalConditionalCFM: Euler ODE solver, 10 steps, cosine t-schedule
//   - HiFTGenerator: F0 prediction → SineGen → ConvTranspose1D chain → iSTFT
//
// Weight loading: reads from chatterbox-s3gen-f16.gguf produced by
// models/convert-chatterbox-to-gguf.py. All tensor names are prefixed
// with "s3." matching the converter's map_s3gen_name().

#include "chatterbox_s3gen.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Context ──────────────────────────────────────────────────────

struct chatterbox_s3gen_context {
    int n_threads = 4;
    int verbosity = 1;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    ~chatterbox_s3gen_context() {
        if (sched) ggml_backend_sched_free(sched);
        if (ctx_w) ggml_free(ctx_w);
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (backend && backend != backend_cpu) ggml_backend_free(backend);
        if (backend_cpu) ggml_backend_free(backend_cpu);
    }
};

// ── Tensor lookup helper ─────────────────────────────────────────

static ggml_tensor* T(chatterbox_s3gen_context* c, const char* name) {
    return core_gguf::try_get(c->tensors, name);
}

static ggml_tensor* TR(chatterbox_s3gen_context* c, const char* name) {
    return core_gguf::require(c->tensors, name, "s3gen");
}

// ── Public API ──────────────────────────────────────────────────

extern "C" struct chatterbox_s3gen_context* chatterbox_s3gen_init_from_file(
    const char* path, int n_threads, int verbosity
) {
    auto* c = new chatterbox_s3gen_context();
    c->n_threads = n_threads > 0 ? n_threads : 4;
    c->verbosity = verbosity;

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "s3gen: failed to init CPU backend\n");
        delete c; return nullptr;
    }
    c->backend = c->backend_cpu;

    // Load weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "s3gen", wl)) {
        delete c; return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (verbosity >= 1) {
        fprintf(stderr, "s3gen: loaded %zu tensors from %s\n", c->tensors.size(), path);
    }

    // Verify critical tensors exist
    if (!TR(c, "s3.flow.input_embedding.weight") ||
        !TR(c, "s3.flow.encoder_proj.weight") ||
        !TR(c, "s3.flow.spk_embed_affine_layer.weight")) {
        fprintf(stderr, "s3gen: missing critical tensors\n");
        delete c; return nullptr;
    }

    // Scheduler
    {
        ggml_backend_t backends[] = { c->backend };
        c->sched = ggml_backend_sched_new(backends, nullptr, 1, 32768, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false));
    }

    return c;
}

// ── Conformer encoder via ggml graph ────────────────────────────
//
// UpsampleConformerEncoder from CosyVoice/ESPnet:
//   embed → pre-lookahead → 6 conformer blocks → upsample 2x →
//   re-embed → 4 conformer blocks → final LayerNorm → project to 80D
//
// Each conformer block:
//   x = x + self_attn(norm_mha(x))   [rel-pos attention, 8 heads]
//   x = x + ffn(norm_ff(x))          [w_1(512→2048) → SiLU → w_2(2048→512)]

// Build one conformer block as ggml ops.
// x: (D, T), returns: (D, T)
static ggml_tensor* build_conformer_block(
    ggml_context* ctx, ggml_cgraph* gf,
    chatterbox_s3gen_context* c,
    ggml_tensor* x, int seq_len, const char* prefix,
    int n_heads, int head_dim, int D, int ff_dim
) {
    const int TT = seq_len; // renamed to avoid shadowing
    char key[64];
    auto W = [&](const char* suffix) -> ggml_tensor* {
        std::snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
        return core_gguf::try_get(c->tensors, key);
    };

    // ---- Self-attention with LayerNorm ----
    ggml_tensor* nmha_w = W("nmha.weight");
    ggml_tensor* nmha_b = W("nmha.bias");

    ggml_tensor* residual = x;
    // LayerNorm
    ggml_tensor* xn = ggml_norm(ctx, x, 1e-5f);
    if (nmha_w) xn = ggml_mul(ctx, xn, nmha_w);
    if (nmha_b) xn = ggml_add(ctx, xn, nmha_b);

    // Q/K/V projections
    ggml_tensor* Q = ggml_mul_mat(ctx, W("sa.lq.weight"), xn);
    ggml_tensor* qb = W("sa.lq.bias");
    if (qb) Q = ggml_add(ctx, Q, qb);
    ggml_tensor* K = ggml_mul_mat(ctx, W("sa.lk.weight"), xn);
    ggml_tensor* kb = W("sa.lk.bias");
    if (kb) K = ggml_add(ctx, K, kb);
    ggml_tensor* V = ggml_mul_mat(ctx, W("sa.lv.weight"), xn);
    ggml_tensor* vb = W("sa.lv.bias");
    if (vb) V = ggml_add(ctx, V, vb);

    // Reshape for multi-head: (D, TT) → (hd, H, TT)
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, TT);
    K = ggml_reshape_3d(ctx, K, head_dim, n_heads, TT);
    V = ggml_reshape_3d(ctx, V, head_dim, n_heads, TT);

    // Simple scaled dot-product attention (without relative position for now)
    // TODO: add relative position encoding with pos_bias_u/v and linear_pos
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // (hd, TT, H)
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    float scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, D, TT);

    // Output projection
    ggml_tensor* attn_out = ggml_mul_mat(ctx, W("sa.lo.weight"), attn);
    ggml_tensor* lo_b = W("sa.lo.bias");
    if (lo_b) attn_out = ggml_add(ctx, attn_out, lo_b);

    x = ggml_add(ctx, residual, attn_out);

    // ---- Feedforward with LayerNorm ----
    residual = x;
    ggml_tensor* nff_w = W("nff.weight");
    ggml_tensor* nff_b = W("nff.bias");
    xn = ggml_norm(ctx, x, 1e-5f);
    if (nff_w) xn = ggml_mul(ctx, xn, nff_w);
    if (nff_b) xn = ggml_add(ctx, xn, nff_b);

    // FFN: w_1 (512→2048) → SiLU → w_2 (2048→512)
    ggml_tensor* ff = ggml_mul_mat(ctx, W("ff.w_1.weight"), xn);
    ggml_tensor* ff_b1 = W("ff.w_1.bias");
    if (ff_b1) ff = ggml_add(ctx, ff, ff_b1);
    ff = ggml_silu(ctx, ff);
    ff = ggml_mul_mat(ctx, W("ff.w_2.weight"), ff);
    ggml_tensor* ff_b2 = W("ff.w_2.bias");
    if (ff_b2) ff = ggml_add(ctx, ff, ff_b2);

    x = ggml_add(ctx, residual, ff);
    return x;
}

// Build the full conformer encoder graph.
// Returns a ggml_cgraph* with "encoder_out" as the output tensor.
static ggml_cgraph* build_graph_conformer_encoder(
    chatterbox_s3gen_context* c, int n_tokens_total
) {
    const int D = 512;
    const int H = 8;
    const int HD = 64;
    const int FF = 2048;
    const int Tin = n_tokens_total;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: token IDs
    ggml_tensor* token_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, Tin);
    ggml_set_name(token_ids, "token_ids");
    ggml_set_input(token_ids);

    // Token embedding lookup
    ggml_tensor* emb_w = TR(c, "s3.flow.input_embedding.weight");
    ggml_tensor* x = ggml_get_rows(ctx0, emb_w, token_ids); // (D, Tin)

    // Linear embed: out.0 (512→512) + LayerNorm out.1
    ggml_tensor* lin_w = T(c, "s3.fe.embed.out.0.weight");
    ggml_tensor* lin_b = T(c, "s3.fe.embed.out.0.bias");
    if (lin_w) {
        x = ggml_mul_mat(ctx0, lin_w, x);
        if (lin_b) x = ggml_add(ctx0, x, lin_b);
    }
    // LayerNorm (embed.out.1)
    ggml_tensor* ln_w = T(c, "s3.fe.embed.out.1.weight");
    ggml_tensor* ln_b = T(c, "s3.fe.embed.out.1.bias");
    if (ln_w) {
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, ln_w);
        if (ln_b) x = ggml_add(ctx0, x, ln_b);
    }

    // Pre-lookahead conv (causal): conv1(k=4, pad_left=3) + LeakyReLU + conv2(k=3, pad_left=2) + residual
    // Skip for now — the causal convs need special handling in ggml
    // TODO: implement pre-lookahead conv

    // 6 conformer blocks (pre-upsample)
    for (int i = 0; i < 6; i++) {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "s3.fe.enc.%d", i);
        x = build_conformer_block(ctx0, gf, c, x, Tin, prefix, H, HD, D, FF);
    }

    // Upsample 2x: interpolate → pad → conv
    // Nearest-neighbor upsample: (D, T) → (D, 2T)
    int T2 = Tin * 2;
    // ggml doesn't have a direct upsample op, so we use repeat
    // Reshape (D, T) → (D, T, 1) → repeat along dim 2 → (D, T, 2) → reshape (D, 2T)
    ggml_tensor* x_3d = ggml_reshape_3d(ctx0, x, D, Tin, 1);
    ggml_tensor* x_up = ggml_repeat_4d(ctx0, x_3d, D, Tin, 2, 1);
    // Interleave: need to transpose the last two dims then flatten
    x_up = ggml_permute(ctx0, x_up, 0, 2, 1, 3); // (D, 2, T)
    x_up = ggml_cont(ctx0, x_up);
    x = ggml_reshape_2d(ctx0, x_up, D, T2);

    // Upsample conv: ul.conv (512, 512, 5) with left-padding
    // For now, skip the upsample conv — just use the interpolated values
    // TODO: implement up_layer.conv

    // Re-embed: up_embed.out.0 (Linear) + up_embed.out.1 (LayerNorm)
    ggml_tensor* uemb_w = T(c, "s3.fe.uemb.out.0.weight");
    ggml_tensor* uemb_b = T(c, "s3.fe.uemb.out.0.bias");
    if (uemb_w) {
        x = ggml_mul_mat(ctx0, uemb_w, x);
        if (uemb_b) x = ggml_add(ctx0, x, uemb_b);
    }
    ggml_tensor* uln_w = T(c, "s3.fe.uemb.out.1.weight");
    ggml_tensor* uln_b = T(c, "s3.fe.uemb.out.1.bias");
    if (uln_w) {
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, uln_w);
        if (uln_b) x = ggml_add(ctx0, x, uln_b);
    }

    // 4 conformer blocks (post-upsample)
    for (int i = 0; i < 4; i++) {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "s3.fe.ue.%d", i);
        x = build_conformer_block(ctx0, gf, c, x, T2, prefix, H, HD, D, FF);
    }

    // Final LayerNorm
    ggml_tensor* an_w = T(c, "s3.fe.an.weight");
    ggml_tensor* an_b = T(c, "s3.fe.an.bias");
    if (an_w) {
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, an_w);
        if (an_b) x = ggml_add(ctx0, x, an_b);
    }

    // Project to 80D: encoder_proj (80, 512)
    ggml_tensor* proj_w = TR(c, "s3.flow.encoder_proj.weight");
    ggml_tensor* proj_b = T(c, "s3.flow.encoder_proj.bias");
    x = ggml_mul_mat(ctx0, proj_w, x);
    if (proj_b) x = ggml_add(ctx0, x, proj_b);

    ggml_set_name(x, "encoder_out");
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);
    return gf;
}

// Run the conformer encoder via ggml graph.
// Returns (80, T_mel) channel-first mel-space encoder output.
static std::vector<float> run_conformer_encoder(
    chatterbox_s3gen_context* c,
    const int32_t* speech_tokens, int n_tokens,
    const int32_t* prompt_tokens, int n_prompt
) {
    const int total = n_prompt + n_tokens;
    const int T_mel = total * 2; // 2x upsample

    // Build token ID array: [prompt | speech]
    std::vector<int32_t> all_tokens(total);
    if (n_prompt > 0) std::memcpy(all_tokens.data(), prompt_tokens, n_prompt * sizeof(int32_t));
    std::memcpy(all_tokens.data() + n_prompt, speech_tokens, n_tokens * sizeof(int32_t));

    // Build and run graph
    ggml_cgraph* gf = build_graph_conformer_encoder(c, total);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "s3gen: failed to alloc conformer graph\n");
        return {};
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "token_ids"),
                            all_tokens.data(), 0, total * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "s3gen: conformer compute failed\n");
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    // out shape: (80, T_mel)
    std::vector<float> h(80 * T_mel);
    ggml_backend_tensor_get(out, h.data(), 0, h.size() * sizeof(float));

    // Convert from (80, T_mel) row-major to (80, T_mel) channel-first
    // ggml stores as ne[0]=80 (fast), ne[1]=T_mel (slow)
    // We need (80, T_mel) where element [ch][t] = h[t * 80 + ch]
    // Transpose to get channel-first
    std::vector<float> h_cf(80 * T_mel);
    for (int t = 0; t < T_mel; t++) {
        for (int ch = 0; ch < 80; ch++) {
            h_cf[ch * T_mel + t] = h[t * 80 + ch];
        }
    }

    return h_cf;
}

// ── Sinusoidal positional embedding ──────────────────────────────

static std::vector<float> sinusoidal_embedding(float t_val, int dim) {
    // Same as SinusoidalPosEmb in matcha/decoder.py
    std::vector<float> emb(dim);
    int half = dim / 2;
    float log_term = std::log(10000.0f) / (float)(half - 1);
    for (int i = 0; i < half; i++) {
        float freq = std::exp(-(float)i * log_term);
        emb[i] = std::sin(t_val * freq);
        emb[half + i] = std::cos(t_val * freq);
    }
    return emb;
}

// ── CFM Euler solver (CPU) ──────────────────────────────────────
//
// For the initial implementation, the denoiser UNet1D forward is a
// stub that returns the mu (encoder output) directly. This produces
// a rough approximation that at least lets the vocoder run.
// TODO: implement full UNet1D denoiser forward.

static std::vector<float> cfm_euler_solve(
    chatterbox_s3gen_context* c,
    const std::vector<float>& mu,       // (80, T) encoder output
    const std::vector<float>& cond,     // (80, T) conditioning (prompt mel + zeros)
    const std::vector<float>& spk_emb,  // (80,) projected speaker embedding
    int T_mel,
    int n_steps,
    float cfg_rate
) {
    // Generate cosine time schedule
    std::vector<float> t_span(n_steps + 1);
    for (int i = 0; i <= n_steps; i++) {
        float t = (float)i / (float)n_steps;
        t_span[i] = 1.0f - std::cos(t * 0.5f * (float)M_PI);
    }

    // Start from noise
    std::vector<float> x(80 * T_mel);
    // Simple noise (deterministic seed for reproducibility)
    uint64_t rng = 42;
    for (size_t i = 0; i < x.size(); i++) {
        // Box-Muller transform for normal distribution
        float u1 = (float)((rng = rng * 6364136223846793005ULL + 1) >> 33) / (float)(1ULL << 31);
        float u2 = (float)((rng = rng * 6364136223846793005ULL + 1) >> 33) / (float)(1ULL << 31);
        if (u1 < 1e-7f) u1 = 1e-7f;
        x[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * (float)M_PI * u2);
    }

    // Euler steps
    for (int step = 0; step < n_steps; step++) {
        float t = t_span[step];
        float r = t_span[step + 1];
        float dt = r - t;

        // TODO: run the actual UNet1D denoiser here.
        // For now, use a simple linear interpolation as placeholder:
        // velocity ≈ (mu - x) which drives x toward mu
        for (size_t i = 0; i < x.size(); i++) {
            float velocity = mu[i] - x[i];
            x[i] += dt * velocity;
        }
    }

    return x; // (80, T_mel)
}

// ── HiFTGenerator vocoder (CPU stub) ────────────────────────────
//
// The full HiFTGenerator has:
//   1. F0 predictor (ConvRNN, 5 conv layers + linear classifier)
//   2. SineGen (harmonic source from F0)
//   3. ConvTranspose1D upsampling (rates 8,5,3 = 120x total)
//   4. Snake activation + ResBlocks
//   5. iSTFT for final waveform
//
// For the initial stub, we use Griffin-Lim as a simple mel→wav
// approximation. The proper HiFTGenerator implementation is a
// follow-up.

static std::vector<float> hift_vocoder_cpu(
    chatterbox_s3gen_context* c,
    const std::vector<float>& mel, // (80, T_mel) channel-first
    int T_mel
) {
    // Mel to waveform via simple Griffin-Lim approximation.
    // The actual HiFTGenerator uses F0-conditioned iSTFT — this is
    // a placeholder that produces audible (but low quality) output.

    const int sample_rate = 24000;
    const int hop_length = 480; // 24000 / 50 Hz mel frame rate
    const int n_samples = T_mel * hop_length;

    std::vector<float> wav(n_samples, 0.0f);

    // Very simple: treat mel as energy envelope, generate noise shaped by it
    for (int t = 0; t < T_mel; t++) {
        // Compute energy from mel bands
        float energy = 0.0f;
        for (int b = 0; b < 80; b++) {
            float m = mel[b * T_mel + t];
            energy += std::exp(m); // mel is in log scale
        }
        energy = std::sqrt(energy / 80.0f) * 0.1f;

        // Fill hop with shaped noise
        for (int s = 0; s < hop_length && (t * hop_length + s) < n_samples; s++) {
            float phase = (float)(t * hop_length + s) / (float)sample_rate;
            // Mix harmonics at common speech frequencies
            wav[t * hop_length + s] = energy * (
                0.5f * std::sin(2.0f * (float)M_PI * 150.0f * phase) +
                0.3f * std::sin(2.0f * (float)M_PI * 300.0f * phase) +
                0.2f * std::sin(2.0f * (float)M_PI * 450.0f * phase)
            );
        }
    }

    return wav;
}

// ── Full pipeline ───────────────────────────────────────────────

extern "C" float* chatterbox_s3gen_synthesize(
    struct chatterbox_s3gen_context* ctx,
    const int32_t* speech_tokens, int n_speech_tokens,
    const int32_t* prompt_tokens, int n_prompt_tokens,
    const float* prompt_feat, int prompt_feat_len,
    const float* spk_embedding,
    int n_cfm_steps,
    int* out_n_samples
) {
    if (!ctx || !speech_tokens || n_speech_tokens <= 0 || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    if (n_cfm_steps <= 0) n_cfm_steps = 10;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: %d speech tokens + %d prompt tokens, %d CFM steps\n",
                n_speech_tokens, n_prompt_tokens, n_cfm_steps);
    }

    // 1. Conformer encoder: tokens → (80, T_mel)
    std::vector<float> h = run_conformer_encoder(
        ctx, speech_tokens, n_speech_tokens,
        prompt_tokens, n_prompt_tokens);

    int T_mel_total = (n_prompt_tokens + n_speech_tokens) * 2; // 2x upsample
    int T_mel_prompt = n_prompt_tokens * 2;
    int T_mel_gen = n_speech_tokens * 2;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: encoder output T_mel=%d (prompt=%d, gen=%d)\n",
                T_mel_total, T_mel_prompt, T_mel_gen);
    }

    // 2. Build conditioning: prompt mel + zeros for generation region
    std::vector<float> cond(80 * T_mel_total, 0.0f);
    if (prompt_feat && prompt_feat_len > 0) {
        int copy_len = std::min(prompt_feat_len, T_mel_prompt);
        // prompt_feat is (T, 80) row-major, convert to (80, T) channel-first
        for (int t = 0; t < copy_len; t++) {
            for (int b = 0; b < 80; b++) {
                cond[b * T_mel_total + t] = prompt_feat[t * 80 + b];
            }
        }
    }

    // 3. Project speaker embedding: spk_embed_affine_layer (80, 192)
    std::vector<float> spk_proj(80, 0.0f);
    if (spk_embedding) {
        ggml_tensor* spk_w = TR(ctx, "s3.flow.spk_embed_affine_layer.weight");
        ggml_tensor* spk_b = T(ctx, "s3.flow.spk_embed_affine_layer.bias");
        std::vector<float> sw(80 * 192);
        std::vector<float> sb(80, 0.0f);
        ggml_backend_tensor_get(spk_w, sw.data(), 0, sw.size() * sizeof(float));
        if (spk_b) ggml_backend_tensor_get(spk_b, sb.data(), 0, sb.size() * sizeof(float));

        // Normalize embedding (L2 norm)
        float norm = 0.0f;
        for (int i = 0; i < 192; i++) norm += spk_embedding[i] * spk_embedding[i];
        norm = std::sqrt(norm + 1e-12f);

        for (int i = 0; i < 80; i++) {
            float sum = sb[i];
            for (int j = 0; j < 192; j++) {
                sum += sw[i * 192 + j] * (spk_embedding[j] / norm);
            }
            spk_proj[i] = sum;
        }
    }

    // 4. CFM Euler solver: noise → mel
    std::vector<float> mel = cfm_euler_solve(
        ctx, h, cond, spk_proj, T_mel_total, n_cfm_steps, 0.7f);

    // 5. Extract generated portion (skip prompt region)
    std::vector<float> gen_mel(80 * T_mel_gen);
    for (int b = 0; b < 80; b++) {
        std::memcpy(&gen_mel[b * T_mel_gen],
                     &mel[b * T_mel_total + T_mel_prompt],
                     T_mel_gen * sizeof(float));
    }

    // 6. Vocoder: mel → waveform
    std::vector<float> wav = hift_vocoder_cpu(ctx, gen_mel, T_mel_gen);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: generated %zu samples (%.2f sec @ 24kHz)\n",
                wav.size(), (float)wav.size() / 24000.0f);
    }

    // Copy to malloc'd buffer
    float* out = (float*)malloc(wav.size() * sizeof(float));
    if (!out) return nullptr;
    std::memcpy(out, wav.data(), wav.size() * sizeof(float));
    *out_n_samples = (int)wav.size();
    return out;
}

extern "C" void chatterbox_s3gen_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" void chatterbox_s3gen_free(struct chatterbox_s3gen_context* ctx) {
    delete ctx;
}
