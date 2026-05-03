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

// ── UNet1D denoiser (ConditionalDecoder) ────────────────────────
//
// The denoiser estimates the velocity field v(x_t, t, conditioning)
// for the flow matching ODE. Architecture:
//   - Time: sinusoidal(320) → MLP(320→1024→1024)
//   - Input: concat [x(80), mu(80), spks_repeat(80), cond(80)] = (320, T)
//   - 1 down block: CausalResnet(320→256) + 4 BasicTransformer(256) + CausalConv(256)
//   - 12 mid blocks: CausalResnet(256→256) + 4 BasicTransformer(256)
//   - 1 up block: CausalResnet(512→256) + 4 BasicTransformer(256) + CausalConv(256)
//   - final: CausalBlock(256→256) + Conv1d(256→80)
//
// For each CFM step, we build a ggml graph for the full denoiser forward.

// Helper: causal conv1d — left-pad by (kernel_size - 1), then conv with padding=0
static ggml_tensor* causal_conv1d(
    ggml_context* ctx, ggml_tensor* x, // input: (T, C_in) in ggml layout
    ggml_tensor* weight, // conv weight: (K, C_in, C_out)
    ggml_tensor* bias    // (C_out) or nullptr
) {
    // Causal conv needs left-padding by (K-1).
    // ggml_pad only pads on the right side.
    // Use symmetric padding in ggml_conv_1d then crop the right side.
    int K = (int)weight->ne[0];
    int pad = K - 1; // total padding we need
    // Use ggml_conv_1d with padding = K-1, then crop K-1 from the right
    ggml_tensor* y = ggml_conv_1d(ctx, weight, x, 1, pad, 1);
    // y has T_out = T + 2*pad - K + 1 = T + pad. Need T, so crop pad from right.
    int T_out = (int)y->ne[0];
    int T_want = (int)x->ne[0];
    if (T_out > T_want) {
        y = ggml_view_2d(ctx, y, T_want, (int)y->ne[1], y->nb[1], 0);
        y = ggml_cont(ctx, y);
    }
    if (bias) {
        // y is (T, C_out), bias is (C_out,) — reshape to (1, C_out) to broadcast
        ggml_tensor* b2d = ggml_reshape_2d(ctx, bias, 1, (int)bias->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// Helper: CausalBlock1D — causal_conv(k=3) + LayerNorm(transpose) + Mish
static ggml_tensor* causal_block1d(
    ggml_context* ctx, ggml_tensor* x, // (C, T)
    ggml_tensor* conv_w, ggml_tensor* conv_b,
    ggml_tensor* ln_w, ggml_tensor* ln_b
) {
    x = causal_conv1d(ctx, x, conv_w, conv_b);
    // LayerNorm over channel dim: transpose (C, T) → (T, C), norm, transpose back
    // After conv1d: x is (T, C) in ggml layout
    // LayerNorm over C dimension (ne[0]=T, ne[1]=C → norm over C)
    // ggml_norm normalizes over ne[0] by default. We need to normalize over C (ne[1]).
    // Transpose to (C, T), norm over C (now ne[0]), transpose back.
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (C, T)
    x = ggml_norm(ctx, x, 1e-5f);
    if (ln_w) x = ggml_mul(ctx, x, ln_w);
    if (ln_b) x = ggml_add(ctx, x, ln_b);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C)
    // SiLU ≈ Mish
    x = ggml_silu(ctx, x);
    return x;
}

// Helper: CausalResnetBlock1D — block1 + time_mlp + block2 + residual
static ggml_tensor* causal_resnet_block(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* t_emb,
    chatterbox_s3gen_context* c, const char* prefix, ggml_tensor* mask
) {
    char key[64];
    auto W = [&](const char* suffix) -> ggml_tensor* {
        std::snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
        return core_gguf::try_get(c->tensors, key);
    };

    ggml_tensor* residual = x;

    // block1: CausalConv1d(k=3) + LayerNorm + Mish
    x = causal_block1d(ctx, x, W("b1.0.weight"), W("b1.0.bias"),
                        W("b1.2.weight"), W("b1.2.bias"));
    // DISABLED: if (mask) x = ggml_mul(ctx, x, mask);

    // Time MLP: linear(1024 → C_out) → add broadcast over T
    // t_emb is (1024,), W is (C_out, 1024) → t_proj is (C_out,)
    ggml_tensor* t_proj = ggml_mul_mat(ctx, W("mlp.1.weight"), t_emb);
    ggml_tensor* t_b = W("mlp.1.bias");
    if (t_b) t_proj = ggml_add(ctx, t_proj, t_b);
    // x is (T, C_out) after causal_block1d. t_proj is (C_out,).
    // Reshape t_proj to (1, C_out) so it broadcasts over T (ne[0]).
    t_proj = ggml_reshape_2d(ctx, t_proj, 1, (int)t_proj->ne[0]);
    // Now (T, C_out) + (1, C_out) → broadcasts T times
    x = ggml_add(ctx, x, t_proj);

    // block2: CausalConv1d(k=3) + LayerNorm + Mish
    x = causal_block1d(ctx, x, W("b2.0.weight"), W("b2.0.bias"),
                        W("b2.2.weight"), W("b2.2.bias"));
    // DISABLED: if (mask) x = ggml_mul(ctx, x, mask);

    // Residual conv (if dimensions differ)
    ggml_tensor* rc_w = W("rc.weight");
    if (rc_w) {
        ggml_tensor* rc_b = W("rc.bias");
        residual = ggml_conv_1d(ctx, rc_w, residual, 1, 0, 1);
        if (rc_b) residual = ggml_add(ctx, residual, ggml_reshape_2d(ctx, rc_b, 1, (int)rc_b->ne[0]));
    }

    return ggml_add(ctx, x, residual);
}

// Helper: BasicTransformerBlock — norm1 → self-attn → norm3 → FF(GEGLU)
static ggml_tensor* basic_transformer_block(
    ggml_context* ctx, ggml_tensor* x, // (C, T) channels-first
    chatterbox_s3gen_context* c, const char* prefix,
    ggml_tensor* attn_mask, // causal mask or nullptr
    int n_heads, int head_dim
) {
    char key[64];
    auto W = [&](const char* suffix) -> ggml_tensor* {
        std::snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
        return core_gguf::try_get(c->tensors, key);
    };

    int C = (int)x->ne[1];
    int TT = (int)x->ne[0];

    // Transpose to (T, C) for attention
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C)

    // norm1 → self-attention
    ggml_tensor* residual = xt;
    ggml_tensor* xn = ggml_norm(ctx, xt, 1e-5f);
    ggml_tensor* n1w = W("norm1.weight");
    ggml_tensor* n1b = W("norm1.bias");
    if (n1w) xn = ggml_mul(ctx, xn, n1w);
    if (n1b) xn = ggml_add(ctx, xn, n1b);

    // Q/K/V projections (no bias on Q/K/V, bias on output)
    ggml_tensor* Q = ggml_mul_mat(ctx, W("attn1.q.weight"), xn);
    ggml_tensor* K = ggml_mul_mat(ctx, W("attn1.k.weight"), xn);
    ggml_tensor* V = ggml_mul_mat(ctx, W("attn1.v.weight"), xn);

    // Multi-head attention: reshape (T, n_heads*2*hd) → (2*hd, T, n_heads)
    // Wait: Q/K/V are (T, 512) where 512 = n_heads(8) * head_dim(64)
    int proj_dim = n_heads * head_dim; // 512
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, TT);
    K = ggml_reshape_3d(ctx, K, head_dim, n_heads, TT);
    V = ggml_reshape_3d(ctx, V, head_dim, n_heads, TT);

    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    float scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, attn_mask, scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, proj_dim, TT); // (T, 512)

    // Output projection
    attn = ggml_mul_mat(ctx, W("attn1.o.weight"), attn);
    ggml_tensor* o_b = W("attn1.o.bias");
    if (o_b) attn = ggml_add(ctx, attn, o_b);

    xt = ggml_add(ctx, residual, attn);

    // norm3 → FF (GEGLU pattern: up_proj with 2x output, split, gate)
    residual = xt;
    xn = ggml_norm(ctx, xt, 1e-5f);
    ggml_tensor* n3w = W("norm3.weight");
    ggml_tensor* n3b = W("norm3.bias");
    if (n3w) xn = ggml_mul(ctx, xn, n3w);
    if (n3b) xn = ggml_add(ctx, xn, n3b);

    // FF up: Linear(C → 4C) + GELU (not GEGLU — decoder uses act_fn="gelu")
    ggml_tensor* ff_up = ggml_mul_mat(ctx, W("ff.up.weight"), xn);
    ggml_tensor* ff_up_b = W("ff.up.bias");
    if (ff_up_b) ff_up = ggml_add(ctx, ff_up, ff_up_b);
    ff_up = ggml_gelu(ctx, ff_up);

    // FF down: Linear(4C → C)
    ggml_tensor* ff_out = ggml_mul_mat(ctx, W("ff.down.weight"), ff_up);
    ggml_tensor* ff_down_b = W("ff.down.bias");
    if (ff_down_b) ff_out = ggml_add(ctx, ff_out, ff_down_b);

    xt = ggml_add(ctx, residual, ff_out);

    // Transpose back to (C, T)
    return ggml_cont(ctx, ggml_transpose(ctx, xt));
}

// Build the full UNet1D denoiser graph for one CFM step.
// x_in: (320, T) = concat[x(80), mu(80), spks(80), cond(80)]
// t_emb: (1024,) time embedding
// Returns: (80, T) velocity prediction
static ggml_cgraph* build_graph_unet1d(
    chatterbox_s3gen_context* c, int T_mel
) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Inputs
    ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, 320);
    ggml_set_name(x_in, "unet_input");
    ggml_set_input(x_in);

    ggml_tensor* t_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1024);
    ggml_set_name(t_emb, "time_emb");
    ggml_set_input(t_emb);

    ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, 1);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    ggml_tensor* x = x_in;

    // ---- Down blocks (1 block) ----
    ggml_tensor* hidden = nullptr;
    {
        x = causal_resnet_block(ctx0, x, t_emb, c, "s3.fd.db.0.0", mask);
        // 4 transformer blocks
        ggml_tensor* xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        for (int j = 0; j < 4; j++) {
            char prefix[48];
            std::snprintf(prefix, sizeof(prefix), "s3.fd.db.0.1.%d", j);
            x = basic_transformer_block(ctx0, x, c, prefix, nullptr, 8, 64);
        }
        hidden = x; // save for skip connection
        // Downsample: CausalConv1d(k=3) — halves T
        ggml_tensor* ds_w = T(c, "s3.fd.db.0.2.weight");
        ggml_tensor* ds_b = T(c, "s3.fd.db.0.2.bias");
        if (ds_w) x = causal_conv1d(ctx0, x, ds_w, ds_b);
        // Note: the Python code uses Downsample1D which actually halves T
        // For CausalConv1d with stride=1, T stays the same
        // The actual downsample uses mask[:, :, ::2] to halve
        // For now keep T unchanged — TODO: proper stride-2 downsample
    }

    // ---- Mid blocks (12 blocks) ----
    for (int i = 0; i < 12; i++) {
        char prefix[48];
        std::snprintf(prefix, sizeof(prefix), "s3.fd.mb.%d.0", i);
        x = causal_resnet_block(ctx0, x, t_emb, c, prefix, mask);

        for (int j = 0; j < 4; j++) {
            char tb_prefix[48];
            std::snprintf(tb_prefix, sizeof(tb_prefix), "s3.fd.mb.%d.1.%d", i, j);
            x = basic_transformer_block(ctx0, x, c, tb_prefix, nullptr, 8, 64);
        }
    }

    // ---- Up blocks (1 block) ----
    {
        // Skip connection: concat with hidden from down block
        if (hidden) {
            // x and hidden should be same size — crop if needed
            int T_x = (int)x->ne[0];
            int T_h = (int)hidden->ne[0];
            if (T_x < T_h) {
                hidden = ggml_view_2d(ctx0, hidden, T_x, (int)hidden->ne[1],
                                      hidden->nb[1], 0);
            }
            x = ggml_concat(ctx0, x, hidden, 1); // concat along channel dim
        }

        x = causal_resnet_block(ctx0, x, t_emb, c, "s3.fd.ub.0.0", mask);

        for (int j = 0; j < 4; j++) {
            char prefix[48];
            std::snprintf(prefix, sizeof(prefix), "s3.fd.ub.0.1.%d", j);
            x = basic_transformer_block(ctx0, x, c, prefix, nullptr, 8, 64);
        }

        // Upsample: CausalConv1d(k=3)
        ggml_tensor* us_w = T(c, "s3.fd.ub.0.2.weight");
        ggml_tensor* us_b = T(c, "s3.fd.ub.0.2.bias");
        if (us_w) x = causal_conv1d(ctx0, x, us_w, us_b);
    }

    // ---- Final block + projection ----
    {
        ggml_tensor* fb_w = T(c, "s3.fd.fb.block.0.weight");
        ggml_tensor* fb_b = T(c, "s3.fd.fb.block.0.bias");
        ggml_tensor* fb_ln_w = T(c, "s3.fd.fb.block.2.weight");
        ggml_tensor* fb_ln_b = T(c, "s3.fd.fb.block.2.bias");
        if (fb_w) x = causal_block1d(ctx0, x, fb_w, fb_b, fb_ln_w, fb_ln_b);
        if (mask) x = ggml_mul(ctx0, x, mask);

        // Final projection: Conv1d(256→80, k=1)
        ggml_tensor* fp_w = T(c, "s3.fd.fp.weight");
        ggml_tensor* fp_b = T(c, "s3.fd.fp.bias");
        if (fp_w) {
            x = ggml_conv_1d(ctx0, fp_w, x, 1, 0, 1);
            if (fp_b) x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, fp_b, 1, (int)fp_b->ne[0]));
        }
    }

    if (mask) x = ggml_mul(ctx0, x, mask);
    ggml_set_name(x, "denoiser_out");
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);
    return gf;
}

// ── CFM Euler solver with UNet1D denoiser ───────────────────────

static std::vector<float> cfm_euler_solve(
    chatterbox_s3gen_context* c,
    const std::vector<float>& mu,       // (80, T) encoder output (channel-first)
    const std::vector<float>& cond,     // (80, T) conditioning mel (channel-first)
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
    uint64_t rng = 42;
    for (size_t i = 0; i < x.size(); i++) {
        float u1 = (float)((rng = rng * 6364136223846793005ULL + 1) >> 33) / (float)(1ULL << 31);
        float u2 = (float)((rng = rng * 6364136223846793005ULL + 1) >> 33) / (float)(1ULL << 31);
        if (u1 < 1e-7f) u1 = 1e-7f;
        x[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * (float)M_PI * u2);
    }

    // Build the UNet1D graph once (can be reused across steps)
    ggml_cgraph* gf = build_graph_unet1d(c, T_mel);

    // Prepare mask (all ones)
    std::vector<float> mask_data(T_mel, 1.0f);

    // Euler ODE steps
    for (int step = 0; step < n_steps; step++) {
        float t_val = t_span[step];
        float r_val = t_span[step + 1];
        float dt = r_val - t_val;

        // Build conditioned input: [x, mu, spks, cond]
        std::vector<float> unet_input(T_mel * 320);
        for (int t = 0; t < T_mel; t++) {
            for (int ch = 0; ch < 80; ch++) {
                unet_input[t * 320 + ch]       = x[ch * T_mel + t];
                unet_input[t * 320 + 80 + ch]  = mu[ch * T_mel + t];
                unet_input[t * 320 + 160 + ch] = spk_emb[ch];
                unet_input[t * 320 + 240 + ch] = cond[ch * T_mel + t];
            }
        }

        // Build unconditioned input for CFG: [x, 0, 0, 0]
        std::vector<float> unet_uncond(T_mel * 320, 0.0f);
        if (cfg_rate > 0.0f) {
            for (int t = 0; t < T_mel; t++) {
                for (int ch = 0; ch < 80; ch++) {
                    unet_uncond[t * 320 + ch] = x[ch * T_mel + t]; // x stays
                    // mu, spks, cond are all zeros (unconditional)
                }
            }
        }

        // Time embedding: sinusoidal(320) → MLP(320→1024→1024)
        std::vector<float> t_sin = sinusoidal_embedding(t_val, 320);
        // MLP on CPU (small, 4 tensors)
        {
            ggml_tensor* tm1_w = T(c, "s3.fd.tm.linear_1.weight");
            ggml_tensor* tm1_b = T(c, "s3.fd.tm.linear_1.bias");
            ggml_tensor* tm2_w = T(c, "s3.fd.tm.linear_2.weight");
            ggml_tensor* tm2_b = T(c, "s3.fd.tm.linear_2.bias");
            if (tm1_w && tm2_w) {
                std::vector<float> w1(1024 * 320), b1(1024, 0.0f);
                std::vector<float> w2(1024 * 1024), b2(1024, 0.0f);
                ggml_backend_tensor_get(tm1_w, w1.data(), 0, w1.size() * sizeof(float));
                if (tm1_b) ggml_backend_tensor_get(tm1_b, b1.data(), 0, b1.size() * sizeof(float));
                ggml_backend_tensor_get(tm2_w, w2.data(), 0, w2.size() * sizeof(float));
                if (tm2_b) ggml_backend_tensor_get(tm2_b, b2.data(), 0, b2.size() * sizeof(float));

                std::vector<float> h1(1024);
                for (int i = 0; i < 1024; i++) {
                    float sum = b1[i];
                    for (int j = 0; j < 320; j++) sum += w1[i * 320 + j] * t_sin[j];
                    h1[i] = sum > 0 ? sum : sum * 0.01f; // SiLU approx
                }
                // SiLU: x * sigmoid(x)
                for (int i = 0; i < 1024; i++) {
                    float sig = 1.0f / (1.0f + std::exp(-h1[i]));
                    h1[i] = h1[i] * sig;
                }
                t_sin.resize(1024);
                for (int i = 0; i < 1024; i++) {
                    float sum = b2[i];
                    for (int j = 0; j < 1024; j++) sum += w2[i * 1024 + j] * h1[j];
                    t_sin[i] = sum;
                }
            }
        }

        // Helper: run denoiser with given input, return velocity
        auto run_denoiser = [&](const std::vector<float>& input) -> std::vector<float> {
            ggml_backend_sched_reset(c->sched);
            if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
                fprintf(stderr, "s3gen: failed to alloc UNet1D graph\n");
                return {};
            }
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "unet_input"),
                                    input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "time_emb"),
                                    t_sin.data(), 0, t_sin.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"),
                                    mask_data.data(), 0, mask_data.size() * sizeof(float));
            if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "s3gen: UNet1D compute failed\n");
                return {};
            }
            ggml_tensor* out = ggml_graph_get_tensor(gf, "denoiser_out");
            if (step == 0 && c->verbosity >= 1) {
                fprintf(stderr, "s3gen: denoiser out ne=(%d, %d)\n",
                        (int)out->ne[0], (int)out->ne[1]);
            }
            size_t nb = ggml_nbytes(out);
            std::vector<float> v(nb / sizeof(float));
            ggml_backend_tensor_get(out, v.data(), 0, nb);
            return v;
        };

        // Run conditioned pass
        std::vector<float> v_cond = run_denoiser(unet_input);
        if (v_cond.empty()) break;

        // Run unconditioned pass for CFG
        std::vector<float> v_uncond;
        if (cfg_rate > 0.0f) {
            v_uncond = run_denoiser(unet_uncond);
        }

        // Read output dimensions from the conditioned pass
        ggml_tensor* out_t = ggml_graph_get_tensor(gf, "denoiser_out");
        int out_T = (int)out_t->ne[0];
        int out_C = (out_t->ne[1] > 1) ? (int)out_t->ne[1] : 1;
        int use_T = std::min(out_T, T_mel);
        int use_C = std::min(out_C, 80);

        // Euler step with CFG blending:
        // v = (1 + cfg_rate) * v_cond - cfg_rate * v_uncond
        for (int ch = 0; ch < use_C; ch++) {
            for (int t = 0; t < use_T; t++) {
                float vc = v_cond[t * out_C + ch];
                float v;
                if (cfg_rate > 0.0f && !v_uncond.empty()) {
                    float vu = v_uncond[t * out_C + ch];
                    v = (1.0f + cfg_rate) * vc - cfg_rate * vu;
                } else {
                    v = vc;
                }
                x[ch * T_mel + t] += dt * v;
            }
        }

        if (c->verbosity >= 2) {
            float rms = 0.0f;
            for (auto v : x) rms += v * v;
            rms = std::sqrt(rms / x.size());
            fprintf(stderr, "s3gen: CFM step %d/%d (t=%.3f→%.3f) x_rms=%.4f\n",
                    step + 1, n_steps, t_val, r_val, rms);
        }
    }

    return x;
}

// ── HiFTGenerator vocoder ────────────────────────────────────────
//
// HiFTNet: Neural Source Filter + iSTFTNet (https://arxiv.org/abs/2309.09493)
//
// Full architecture:
//   1. F0 predictor: 5× Conv1d(k=3,p=1) → ELU + Linear → |F0|
//   2. SineGen: F0 → harmonic source waveform
//   3. conv_pre(80→512,k=7) → 3 upsample stages (ConvTranspose1d + ResBlocks)
//   4. Source fusion at each stage (STFT of source → down-conv → add)
//   5. conv_post(64→18,k=7) → split to magnitude(9) + phase(9) → iSTFT
//
// Current implementation: F0 predictor (real weights) + simplified iSTFT.
// The full ConvTranspose1d + ResBlock + Snake chain is a follow-up.

// Run F0 predictor: mel (T, 80) → F0 (T,)
static std::vector<float> run_f0_predictor(
    chatterbox_s3gen_context* c,
    const std::vector<float>& mel, // (80, T_mel) channel-first
    int T_mel
) {
    // F0 predictor: 5× Conv1d(80→512, k=3, p=1) + ELU, then Linear(512→1) → abs()
    // Run on CPU since it's small
    const int C = 512;
    const int K = 3;

    // Convert mel to (T, 80) row-major for Conv1d processing
    std::vector<float> x(T_mel * 80);
    for (int t = 0; t < T_mel; t++)
        for (int c2 = 0; c2 < 80; c2++)
            x[t * 80 + c2] = mel[c2 * T_mel + t];

    // 5 conv layers
    for (int layer = 0; layer < 5; layer++) {
        char wn[48], bn[48];
        std::snprintf(wn, sizeof(wn), "s3.v.f0.cn.%d.weight", layer * 2);
        std::snprintf(bn, sizeof(bn), "s3.v.f0.cn.%d.bias", layer * 2);
        ggml_tensor* wt = T(c, wn);
        ggml_tensor* bt = T(c, bn);
        if (!wt) continue;

        int C_in = (layer == 0) ? 80 : C;
        int C_out = C;

        // Read weights, handling F16→F32 conversion
        size_t n_elem = (size_t)K * C_in * C_out;
        std::vector<float> w_f32(n_elem);
        std::vector<float> b(C_out, 0.0f);
        if (wt->type == GGML_TYPE_F16) {
            std::vector<char> raw(ggml_nbytes(wt));
            ggml_backend_tensor_get(wt, raw.data(), 0, raw.size());
            const ggml_fp16_t* w16 = (const ggml_fp16_t*)raw.data();
            for (size_t i = 0; i < n_elem; i++)
                w_f32[i] = ggml_fp16_to_fp32(w16[i]);
        } else {
            ggml_backend_tensor_get(wt, w_f32.data(), 0, n_elem * sizeof(float));
        }
        if (bt) ggml_backend_tensor_get(bt, b.data(), 0, C_out * sizeof(float));

        // Conv1d with padding=1 (symmetric): out[t] = sum over k,c_in
        // Input x is (T, C_in), weight is (C_out, C_in, K) in memory
        // PyTorch Conv1d: out[co, t] = bias[co] + sum_ci sum_k w[co,ci,k] * x[ci, t+k-pad]
        // Our layout: x[t * C_in + ci], w[co * C_in * K + ci * K + k]
        std::vector<float> out(T_mel * C_out, 0.0f);
        for (int t = 0; t < T_mel; t++) {
            for (int co = 0; co < C_out; co++) {
                float sum = b[co];
                for (int k = 0; k < K; k++) {
                    int tt = t + k - 1; // padding=1
                    if (tt < 0 || tt >= T_mel) continue;
                    for (int ci = 0; ci < C_in; ci++) {
                        // w layout: (K, C_in, C_out) → w[k * C_in * C_out + ci * C_out + co]
                        // Actually ggml stores as ne[0]=K, ne[1]=C_in, ne[2]=C_out
                        // Memory: w[co * C_in * K + ci * K + k]
                        sum += w_f32[co * C_in * K + ci * K + k] * x[tt * C_in + ci];
                    }
                }
                // ELU activation
                if (sum < 0) sum = std::exp(sum) - 1.0f;
                out[t * C_out + co] = sum;
            }
        }
        x = std::move(out);
    }

    // Linear classifier: (512 → 1) + abs
    ggml_tensor* cls_w = T(c, "s3.v.f0.cls.weight");
    ggml_tensor* cls_b = T(c, "s3.v.f0.cls.bias");
    std::vector<float> f0(T_mel, 0.0f);
    if (cls_w) {
        std::vector<float> cw_f32(C);
        if (cls_w->type == GGML_TYPE_F16) {
            std::vector<char> raw(ggml_nbytes(cls_w));
            ggml_backend_tensor_get(cls_w, raw.data(), 0, raw.size());
            const ggml_fp16_t* cw16 = (const ggml_fp16_t*)raw.data();
            for (int i = 0; i < C; i++) cw_f32[i] = ggml_fp16_to_fp32(cw16[i]);
        } else {
            ggml_backend_tensor_get(cls_w, cw_f32.data(), 0, C * sizeof(float));
        }
        float cb = 0.0f;
        if (cls_b) ggml_backend_tensor_get(cls_b, &cb, 0, sizeof(float));

        for (int t = 0; t < T_mel; t++) {
            float sum = cb;
            for (int i = 0; i < C; i++) sum += cw_f32[i] * x[t * C + i];
            f0[t] = std::abs(sum);
        }
    }

    return f0;
}

// HiFTGenerator vocoder: mel (80, T) → waveform via learned upsampling + iSTFT
//
// Architecture: conv_pre(80→512) → 3 upsample stages with ConvTranspose1d
// → conv_post(64→18) → split magnitude(9)/phase(9) → iSTFT(n_fft=16, hop=4)
//
// Each upsample stage: LeakyReLU → ConvTranspose1d(↑) → source_fusion → 3 ResBlocks
// Total upsample factor: 8 × 5 × 3 = 120, then iSTFT hop=4 → 480 samples/mel_frame
//
// For now: simplified path using conv_pre + conv_post + iSTFT, skipping
// the intermediate ResBlocks and source fusion. This produces usable
// (if noisy) audio because the learned conv_pre/post capture the mel→wav mapping.

static std::vector<float> hift_vocoder_cpu(
    chatterbox_s3gen_context* c,
    const std::vector<float>& mel, // (80, T_mel) channel-first
    int T_mel
) {
    if (c->verbosity >= 1) {
        float mel_rms = 0, mel_max = 0;
        for (size_t i = 0; i < mel.size(); i++) {
            mel_rms += mel[i] * mel[i];
            if (std::abs(mel[i]) > mel_max) mel_max = std::abs(mel[i]);
        }
        mel_rms = std::sqrt(mel_rms / mel.size());
        fprintf(stderr, "s3gen: vocoder mel T=%d rms=%.3f max=%.3f\n", T_mel, mel_rms, mel_max);
    }

    // Build and run HiFTGenerator ggml graph:
    // conv_pre(80→512,k=7) → 3× [LeakyReLU → ConvTranspose1d(↑) → skip ResBlocks]
    // → LeakyReLU → conv_post(64→18,k=7) → split mag(9)/phase(9) → iSTFT
    const int istft_nfft = 16;
    const int istft_hop = 4;

    // Build graph
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input: mel (T_mel, 80) in ggml layout
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, 80);
    ggml_set_name(x, "voc_mel");
    ggml_set_input(x);

    // conv_pre: Conv1d(80→512, k=7, padding=3)
    ggml_tensor* cpre_w = T(c, "s3.v.cpre.weight");
    ggml_tensor* cpre_b = T(c, "s3.v.cpre.bias");
    if (cpre_w) {
        x = ggml_conv_1d(ctx0, cpre_w, x, 1, 3, 1);
        if (cpre_b) x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, cpre_b, 1, (int)cpre_b->ne[0]));
    }

    // Source STFT input (from SineGen → STFT): 18 channels (real + imag)
    // Total audio length = T_mel * 120 (upsample) * 4 (iSTFT hop)
    // Source operates at audio rate: T_audio = T_mel * upsample_total * istft_hop
    // STFT of source: n_frames ≈ T_audio
    // For simplicity, we compute source length from T_mel and provide it as input
    int T_audio = T_mel * 120 * 4; // total audio samples
    // Source STFT: after STFT(n_fft=16, hop=4), n_frames ≈ T_audio / hop ≈ T_mel * 120
    int T_src = T_mel * 120;
    ggml_tensor* s_stft = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_src, 18);
    ggml_set_name(s_stft, "source_stft");
    ggml_set_input(s_stft);

    // 3 upsample stages
    const int strides[] = {8, 5, 3};
    const int kernels[] = {16, 11, 7};
    for (int stage = 0; stage < 3; stage++) {
        // LeakyReLU(0.1)
        x = ggml_leaky_relu(ctx0, x, 0.1f, false);

        // ConvTranspose1d
        char wn[32], bn[32];
        std::snprintf(wn, sizeof(wn), "s3.v.ups.%d.weight", stage);
        std::snprintf(bn, sizeof(bn), "s3.v.ups.%d.bias", stage);
        ggml_tensor* up_w = T(c, wn);
        ggml_tensor* up_b = T(c, bn);
        if (up_w) {
            int s = strides[stage];
            int k = kernels[stage];
            int p = (k - s) / 2;
            // ggml_conv_transpose_1d doesn't support non-zero padding
            // Run with p=0, then crop (k-s)/2 from each side
            int T_in = (int)x->ne[0];
            x = ggml_conv_transpose_1d(ctx0, up_w, x, s, 0, 1);
            // Output length with p=0: (T_in - 1) * s + k
            // Expected with p: (T_in - 1) * s + k - 2*p = T_in * s
            if (p > 0) {
                int T_out = (int)((T_in - 1) * s + k);
                int T_want = T_in * s; // expected output length
                int C_out = (int)x->ne[1];
                // Crop p from left: view starting at offset p
                x = ggml_view_2d(ctx0, x, T_want, C_out, x->nb[1], p * x->nb[0]);
                x = ggml_cont(ctx0, x);
            }
            if (up_b) x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, up_b, 1, (int)up_b->ne[0]));
        }

        // Source fusion: source_downs[i](s_stft) → source_resblocks[i] → add
        // The source STFT comes from SineGen(F0). Since F0≈0 (unvoiced),
        // the source is noise. We provide the noise STFT as a graph input.
        {
            char sd_wn[32], sd_bn[32];
            std::snprintf(sd_wn, sizeof(sd_wn), "s3.v.sd.%d.weight", stage);
            std::snprintf(sd_bn, sizeof(sd_bn), "s3.v.sd.%d.bias", stage);
            ggml_tensor* sd_w = T(c, sd_wn);
            ggml_tensor* sd_b = T(c, sd_bn);
            if (sd_w && s_stft) {
                // source_downs: Conv1d that downsamples s_stft to match x's time dim
                ggml_tensor* si = ggml_conv_1d(ctx0, sd_w, s_stft, 1, 0, 1);
                if (sd_b) si = ggml_add(ctx0, si, ggml_reshape_2d(ctx0, sd_b, 1, (int)sd_b->ne[0]));
                // source_resblocks: same structure as main ResBlocks but with source-specific weights
                // Use the same Snake + dilated conv pattern
                const int srb_kernels[] = {7, 7, 11};
                const int srb_dilations[][3] = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
                ggml_tensor* srb_in = si;
                for (int d = 0; d < 3; d++) {
                    char key2[48];
                    int dil = srb_dilations[stage][d];
                    int k2 = srb_kernels[stage];
                    int pad2 = (k2 * dil - dil) / 2;
                    // Snake1
                    std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.a1.%d.alpha", stage, d);
                    ggml_tensor* sa1 = T(c, key2);
                    if (sa1) {
                        ggml_tensor* a = ggml_reshape_2d(ctx0, sa1, 1, (int)sa1->ne[0]);
                        ggml_tensor* ax = ggml_mul(ctx0, si, a);
                        ggml_tensor* s_ax = ggml_sin(ctx0, ax);
                        si = ggml_add(ctx0, si, ggml_div(ctx0, ggml_mul(ctx0, s_ax, s_ax), a));
                    }
                    // Conv1
                    std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.c1.%d.weight", stage, d);
                    ggml_tensor* sc1w = T(c, key2);
                    std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.c1.%d.bias", stage, d);
                    ggml_tensor* sc1b = T(c, key2);
                    if (sc1w) {
                        si = ggml_conv_1d(ctx0, sc1w, si, 1, pad2, dil);
                        if (sc1b) si = ggml_add(ctx0, si, ggml_reshape_2d(ctx0, sc1b, 1, (int)sc1b->ne[0]));
                    }
                    // Snake2
                    std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.a2.%d.alpha", stage, d);
                    ggml_tensor* sa2 = T(c, key2);
                    if (sa2) {
                        ggml_tensor* a2 = ggml_reshape_2d(ctx0, sa2, 1, (int)sa2->ne[0]);
                        ggml_tensor* ax2 = ggml_mul(ctx0, si, a2);
                        ggml_tensor* s_ax2 = ggml_sin(ctx0, ax2);
                        si = ggml_add(ctx0, si, ggml_div(ctx0, ggml_mul(ctx0, s_ax2, s_ax2), a2));
                    }
                    // Conv2
                    std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.c2.%d.weight", stage, d);
                    ggml_tensor* sc2w = T(c, key2);
                    std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.c2.%d.bias", stage, d);
                    ggml_tensor* sc2b = T(c, key2);
                    if (sc2w) {
                        int p2 = (k2 - 1) / 2;
                        si = ggml_conv_1d(ctx0, sc2w, si, 1, p2, 1);
                        if (sc2b) si = ggml_add(ctx0, si, ggml_reshape_2d(ctx0, sc2b, 1, (int)sc2b->ne[0]));
                    }
                    si = ggml_add(ctx0, si, srb_in);
                    srb_in = si;
                }
                // Crop si to match x's time dimension and add
                int T_x = (int)x->ne[0];
                int T_si = (int)si->ne[0];
                if (T_si > T_x) {
                    si = ggml_view_2d(ctx0, si, T_x, (int)si->ne[1], si->nb[1], 0);
                    si = ggml_cont(ctx0, si);
                }
                x = ggml_add(ctx0, x, si);
            }
        }

        // ResBlocks: 3 per stage, each run INDEPENDENTLY on the same input,
        // then outputs averaged: x = (rb0(x) + rb1(x) + rb2(x)) / 3
        const int rb_kernels[] = {3, 7, 11};
        const int rb_dilations[][3] = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
        ggml_tensor* rb_sum = nullptr;
        ggml_tensor* rb_input = x; // save input for each independent ResBlock
        for (int rb = 0; rb < 3; rb++) {
            x = rb_input; // reset to same input for each ResBlock
            int rb_idx = stage * 3 + rb;
            // ResBlock: for each of 3 dilated passes: snake1 → conv1(dilated) → snake2 → conv2 → residual
            ggml_tensor* rb_residual = x;
            for (int d = 0; d < 3; d++) {
                char key[48];
                int dil = rb_dilations[rb][d];
                int k = rb_kernels[rb];
                int pad = (k * dil - dil) / 2; // get_padding(k, dil)

                // Snake activation 1: x + (1/alpha) * sin²(alpha * x)
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.a1.%d.alpha", rb_idx, d);
                ggml_tensor* alpha1 = T(c, key);
                if (alpha1) {
                    // Snake: x + (1/alpha) * sin²(alpha * x)
                    ggml_tensor* a = ggml_reshape_2d(ctx0, alpha1, 1, (int)alpha1->ne[0]);
                    ggml_tensor* ax = ggml_mul(ctx0, x, a);
                    ggml_tensor* sin_ax = ggml_sin(ctx0, ax);
                    ggml_tensor* sin2 = ggml_mul(ctx0, sin_ax, sin_ax);
                    // sin²(ax) / alpha
                    ggml_tensor* sin2_over_a = ggml_div(ctx0, sin2, a);
                    x = ggml_add(ctx0, x, sin2_over_a);
                }

                // Conv1d with dilation
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.c1.%d.weight", rb_idx, d);
                ggml_tensor* c1w = T(c, key);
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.c1.%d.bias", rb_idx, d);
                ggml_tensor* c1b = T(c, key);
                if (c1w) {
                    x = ggml_conv_1d(ctx0, c1w, x, 1, pad, dil);
                    if (c1b) x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c1b, 1, (int)c1b->ne[0]));
                }

                // Snake activation 2: same as activation 1
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.a2.%d.alpha", rb_idx, d);
                ggml_tensor* alpha2 = T(c, key);
                if (alpha2) {
                    ggml_tensor* a2 = ggml_reshape_2d(ctx0, alpha2, 1, (int)alpha2->ne[0]);
                    ggml_tensor* ax2 = ggml_mul(ctx0, x, a2);
                    ggml_tensor* sin_ax2 = ggml_sin(ctx0, ax2);
                    ggml_tensor* sin2_2 = ggml_mul(ctx0, sin_ax2, sin_ax2);
                    ggml_tensor* sin2_over_a2 = ggml_div(ctx0, sin2_2, a2);
                    x = ggml_add(ctx0, x, sin2_over_a2);
                }

                // Conv2 (dilation=1)
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.c2.%d.weight", rb_idx, d);
                ggml_tensor* c2w = T(c, key);
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.c2.%d.bias", rb_idx, d);
                ggml_tensor* c2b = T(c, key);
                if (c2w) {
                    int pad2 = (k - 1) / 2; // dilation=1, symmetric padding
                    x = ggml_conv_1d(ctx0, c2w, x, 1, pad2, 1);
                    if (c2b) x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c2b, 1, (int)c2b->ne[0]));
                }

                // Residual
                x = ggml_add(ctx0, x, rb_residual);
                rb_residual = x;
            }
            // Accumulate for averaging
            if (!rb_sum) rb_sum = x;
            else rb_sum = ggml_add(ctx0, rb_sum, x);
        }
        // Average the 3 ResBlock outputs
        x = ggml_scale(ctx0, rb_sum, 1.0f / 3.0f);
    }

    // Reflection pad (1, 0) — skip for now
    // LeakyReLU
    x = ggml_leaky_relu(ctx0, x, 0.1f, false);

    // conv_post: Conv1d(64→18, k=7, padding=3)
    ggml_tensor* cpost_w = T(c, "s3.v.cpost.weight");
    ggml_tensor* cpost_b = T(c, "s3.v.cpost.bias");
    if (cpost_w) {
        x = ggml_conv_1d(ctx0, cpost_w, x, 1, 3, 1);
        if (cpost_b) x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, cpost_b, 1, (int)cpost_b->ne[0]));
    }

    // Clamp to prevent iSTFT overflow from unbounded ResBlock output
    x = ggml_clamp(ctx0, x, -5.0f, 5.0f);

    ggml_set_name(x, "voc_stft");
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);

    // Execute graph
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "s3gen: failed to alloc vocoder graph\n");
        // Fallback to noise
        return std::vector<float>(T_mel * 480, 0.0f);
    }

    // Set mel input (convert from channel-first to ggml (T, C))
    std::vector<float> mel_tf(T_mel * 80);
    for (int t = 0; t < T_mel; t++)
        for (int b = 0; b < 80; b++)
            mel_tf[t * 80 + b] = mel[b * T_mel + t];
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "voc_mel"),
                            mel_tf.data(), 0, mel_tf.size() * sizeof(float));

    // Generate source STFT: STFT of noise source (since F0≈0, source is noise)
    {
    std::vector<float> src_stft(T_src * 18, 0.0f);
    {
        // Generate noise source
        float nsf_alpha = 0.1f; // SineGen sine_amp
        uint64_t rng = 54321;
        for (int t = 0; t < T_src; t++) {
            // For unvoiced (F0=0): noise only, amplitude = nsf_alpha/3
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            float noise = ((float)(int64_t)(rng >> 33) / (float)(1LL << 30)) - 1.0f;
            float src = noise * nsf_alpha / 3.0f;
            // Simple STFT: for frame t, compute real/imag of N/2+1 bins
            // Since we're operating frame-by-frame with hop=1 here, just use
            // the instantaneous amplitude as the DC component
            src_stft[t * 18 + 0] = src;  // real[0] (DC)
            // Higher frequency bins get lower amplitude
            for (int f = 1; f < 9; f++) {
                rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                float phase = ((float)(rng >> 33) / (float)(1LL << 31)) * 2.0f * (float)M_PI;
                src_stft[t * 18 + f] = src * std::cos(phase) * 0.5f;     // real[f]
                src_stft[t * 18 + 9 + f] = src * std::sin(phase) * 0.5f; // imag[f]
            }
        }
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "source_stft"),
                            src_stft.data(), 0, src_stft.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "s3gen: vocoder compute failed\n");
        return std::vector<float>(T_mel * 480, 0.0f);
    }

    // Read STFT output
    ggml_tensor* stft_out = ggml_graph_get_tensor(gf, "voc_stft");
    int T_stft = (int)stft_out->ne[0];
    int C_stft = (int)stft_out->ne[1]; // should be 18
    if (c->verbosity >= 1) {
        fprintf(stderr, "s3gen: vocoder STFT output (%d, %d)\n", T_stft, C_stft);
    }

    std::vector<float> stft_data(T_stft * C_stft);
    ggml_backend_tensor_get(stft_out, stft_data.data(), 0, ggml_nbytes(stft_out));

    // Diagnostic: check STFT output statistics
    if (c->verbosity >= 1) {
        float stft_rms = 0, stft_max = 0, stft_min = 1e30f;
        for (size_t i = 0; i < stft_data.size(); i++) {
            stft_rms += stft_data[i] * stft_data[i];
            if (stft_data[i] > stft_max) stft_max = stft_data[i];
            if (stft_data[i] < stft_min) stft_min = stft_data[i];
        }
        stft_rms = std::sqrt(stft_rms / stft_data.size());
        fprintf(stderr, "s3gen: STFT values range=[%.3f, %.3f] rms=%.4f (ref: [-1.1, 1.7])\n",
                stft_min, stft_max, stft_rms);
        // Show first frame's 18 channels
        fprintf(stderr, "s3gen: STFT frame[0]: ");
        for (int ch = 0; ch < C_stft && ch < 18; ch++)
            fprintf(stderr, "%.3f ", stft_data[ch]);
        fprintf(stderr, "\n");
    }

    // iSTFT: split into magnitude (first 9 channels) and phase (last 9 channels)
    // Python: magnitude = exp(clip(x[:, :9, :], max=100))
    //         phase = sin(x[:, 9:, :])
    //         complex = magnitude * (cos(phase) + j * sin(phase))
    //         wav = torch.istft(complex, n_fft=16, hop_length=4, win_length=16, window=hann)
    int n_freq = istft_nfft / 2 + 1; // 9
    int n_samples = (T_stft - 1) * istft_hop + istft_nfft;

    std::vector<float> wav(n_samples, 0.0f);
    std::vector<float> win_sum(n_samples, 0.0f);

    // Hann window
    std::vector<float> win(istft_nfft);
    for (int i = 0; i < istft_nfft; i++)
        win[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / (float)istft_nfft));

    for (int frame = 0; frame < T_stft; frame++) {
        // stft_data layout: (T_stft, 18) — first 9 = log-magnitude, last 9 = raw phase input
        // magnitude = exp(clip(raw, max=100))
        // phase = sin(raw_phase)  — "actually, sin is redundancy" per Python comment
        float mag[9], ph[9];
        for (int f = 0; f < n_freq; f++) {
            float raw_mag = stft_data[frame * C_stft + f];
            if (raw_mag > 100.0f) raw_mag = 100.0f; // clip
            mag[f] = std::exp(raw_mag);
            ph[f] = std::sin(stft_data[frame * C_stft + n_freq + f]);
        }

        // Build complex STFT frame: real = mag * cos(phase), imag = mag * sin(phase)
        // Then iDFT: x[n] = (1/N) * sum_k (real[k]*cos(2πkn/N) - imag[k]*sin(2πkn/N))
        // For real signal: use Hermitian symmetry
        int start = frame * istft_hop;
        for (int n = 0; n < istft_nfft && (start + n) < n_samples; n++) {
            float sample = 0.0f;

            // DC component (f=0)
            sample += mag[0] * std::cos(ph[0]);

            // Positive frequencies + their conjugate (Hermitian)
            for (int f = 1; f < n_freq - 1; f++) {
                float real_f = mag[f] * std::cos(ph[f]);
                float imag_f = mag[f] * std::sin(ph[f]);
                float angle = 2.0f * (float)M_PI * f * n / (float)istft_nfft;
                // Positive frequency + conjugate = 2 * Re(X[f] * e^{j*angle})
                sample += 2.0f * (real_f * std::cos(angle) - imag_f * std::sin(angle));
            }

            // Nyquist (f=n_freq-1 = N/2)
            sample += mag[n_freq - 1] * std::cos(ph[n_freq - 1]) *
                      std::cos(2.0f * (float)M_PI * (n_freq - 1) * n / (float)istft_nfft);

            sample /= (float)istft_nfft;

            // Overlap-add with Hann window
            wav[start + n] += sample * win[n];
            win_sum[start + n] += win[n] * win[n];
        }
    }

    // Normalize by COLA (constant overlap-add) condition
    for (int i = 0; i < n_samples; i++) {
        if (win_sum[i] > 1e-8f) wav[i] /= win_sum[i];
    }

    // Clamp to [-0.99, 0.99]
    for (float& v : wav) v = std::max(-0.99f, std::min(0.99f, v));

    // Trim to expected length
    int final_len = T_mel * 480;
    if ((int)wav.size() > final_len) wav.resize(final_len);
    else if ((int)wav.size() < final_len) wav.resize(final_len, 0.0f);

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
