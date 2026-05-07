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

// Read tensor data into float buffer, dequantizing if needed.
static void tensor_get_f32(ggml_tensor* t, float* out, size_t offset_bytes, size_t n_elem) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out, offset_bytes, n_elem * sizeof(float));
    } else {
        size_t raw_bytes = ggml_nbytes(t);
        if (offset_bytes > 0) {
            // Partial read: dequantize entire tensor then copy the slice
            std::vector<char> raw(raw_bytes);
            ggml_backend_tensor_get(t, raw.data(), 0, raw_bytes);
            size_t total_elem = ggml_nelements(t);
            std::vector<float> all(total_elem);
            const auto* tt = ggml_get_type_traits(t->type);
            if (tt && tt->to_float) {
                tt->to_float(raw.data(), all.data(), (int64_t)total_elem);
            } else {
                std::memset(all.data(), 0, total_elem * sizeof(float));
            }
            size_t start_elem = offset_bytes / sizeof(float);
            std::memcpy(out, all.data() + start_elem, n_elem * sizeof(float));
        } else {
            std::vector<char> raw(raw_bytes);
            ggml_backend_tensor_get(t, raw.data(), 0, raw_bytes);
            const auto* tt = ggml_get_type_traits(t->type);
            if (tt && tt->to_float) {
                tt->to_float(raw.data(), out, (int64_t)n_elem);
            } else {
                std::memset(out, 0, n_elem * sizeof(float));
            }
        }
    }
}

// ── Context ──────────────────────────────────────────────────────

struct chatterbox_s3gen_context {
    int n_threads = 4;
    int verbosity = 1;
    bool istft_full_idft = false; // CRISPASR_HIFT_FULL_IDFT=1 → torch.istft-compatible path

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    ~chatterbox_s3gen_context() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (ctx_w)
            ggml_free(ctx_w);
        if (buf_w)
            ggml_backend_buffer_free(buf_w);
        if (backend && backend != backend_cpu)
            ggml_backend_free(backend);
        if (backend_cpu)
            ggml_backend_free(backend_cpu);
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

extern "C" struct chatterbox_s3gen_context* chatterbox_s3gen_init_from_file(const char* path, int n_threads,
                                                                            int verbosity, bool use_gpu) {
    auto* c = new chatterbox_s3gen_context();
    c->n_threads = n_threads > 0 ? n_threads : 4;
    c->verbosity = verbosity;
    {
        const char* env = std::getenv("CRISPASR_HIFT_FULL_IDFT");
        c->istft_full_idft = env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
    }

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "s3gen: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    c->backend = use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        if (verbosity >= 1 && use_gpu) {
            fprintf(stderr, "s3gen: GPU backend unavailable, falling back to CPU\n");
        }
        c->backend = c->backend_cpu;
    }

    // Load weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "s3gen", wl)) {
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (verbosity >= 1) {
        fprintf(stderr, "s3gen: loaded %zu tensors from %s\n", c->tensors.size(), path);
    }

    // Verify critical tensors exist
    if (!TR(c, "s3.flow.input_embedding.weight") || !TR(c, "s3.flow.encoder_proj.weight") ||
        !TR(c, "s3.flow.spk_embed_affine_layer.weight")) {
        fprintf(stderr, "s3gen: missing critical tensors\n");
        delete c;
        return nullptr;
    }

    // Scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = c->backend;
        if (c->backend != c->backend_cpu)
            backends[n_be++] = c->backend_cpu;
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
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
static ggml_tensor* build_conformer_block(ggml_context* ctx, ggml_cgraph* gf, chatterbox_s3gen_context* c,
                                          ggml_tensor* x, int seq_len, const char* prefix, int n_heads, int head_dim,
                                          int D, int ff_dim, ggml_tensor* pos_emb = nullptr) {
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
    if (nmha_w)
        xn = ggml_mul(ctx, xn, nmha_w);
    if (nmha_b)
        xn = ggml_add(ctx, xn, nmha_b);

    // Mark pre-attention norm for first block dump
    if (std::strcmp(prefix + std::strlen(prefix) - 2, ".0") == 0 && std::strstr(prefix, "enc.")) {
        ggml_set_name(xn, "dump_enc0_pre_attn_norm");
        ggml_set_output(xn);
    }

    // Q/K/V projections: (D, TT) → (D, TT)
    ggml_tensor* Q = ggml_mul_mat(ctx, W("sa.lq.weight"), xn);
    ggml_tensor* qb = W("sa.lq.bias");
    if (qb)
        Q = ggml_add(ctx, Q, qb);
    ggml_tensor* K = ggml_mul_mat(ctx, W("sa.lk.weight"), xn);
    ggml_tensor* kb = W("sa.lk.bias");
    if (kb)
        K = ggml_add(ctx, K, kb);
    ggml_tensor* V = ggml_mul_mat(ctx, W("sa.lv.weight"), xn);
    ggml_tensor* vb = W("sa.lv.bias");
    if (vb)
        V = ggml_add(ctx, V, vb);

    // Reshape for multi-head: (D, TT) → (hd, H, TT)
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, TT);
    K = ggml_reshape_3d(ctx, K, head_dim, n_heads, TT);
    V = ggml_reshape_3d(ctx, V, head_dim, n_heads, TT);

    // Check for relative position attention weights
    ggml_tensor* pos_bias_u = W("sa.pbu");
    ggml_tensor* pos_bias_v = W("sa.pbv");
    ggml_tensor* linear_pos_w = W("sa.lp.weight");

    ggml_tensor* attn;
    if (pos_bias_u && pos_bias_v && linear_pos_w && pos_emb) {
        // Relative position multi-headed attention (Transformer-XL / ESPnet style)
        // Q is (hd, H, TT). Transpose to (hd, TT, H) for matmul.
        ggml_tensor* q_t = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // (hd, TT, H)
        ggml_tensor* k_t = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        // Position encoding: p = linear_pos(pos_emb) → reshape to (hd, T_pos, H)
        // pos_emb: (D, T_pos) where T_pos = 2*TT-1 for Espnet relative encoding
        ggml_tensor* p = ggml_mul_mat(ctx, linear_pos_w, pos_emb); // (D, T_pos)
        int T_pos = (int)pos_emb->ne[1];
        p = ggml_reshape_3d(ctx, p, head_dim, n_heads, T_pos);
        p = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3)); // (hd, T_pos, H)

        // Dump linear_pos output for enc.0
        if (std::strcmp(prefix + std::strlen(prefix) - 2, ".0") == 0 && std::strstr(prefix, "enc.")) {
            ggml_tensor* p_dump = ggml_scale(ctx, p, 1.0f);
            ggml_set_name(p_dump, "dump_enc0_linear_pos_out");
            ggml_set_output(p_dump);
            ggml_build_forward_expand(gf, p_dump);
        }

        // pos_bias_u/v: stored as (hd, H) in GGUF (ne[0]=hd, ne[1]=H).
        // Element (d, h) = PyTorch pbu[h][d]. Just reshape to (hd, 1, H)
        // for broadcast add — NO transpose (transpose scrambles head/dim).
        // Cast to F32 if stored as F16 (base chatterbox S3Gen uses F16).
        ggml_tensor* pbu_src = pos_bias_u;
        ggml_tensor* pbv_src = pos_bias_v;
        if (pbu_src->type != GGML_TYPE_F32)
            pbu_src = ggml_cast(ctx, pbu_src, GGML_TYPE_F32);
        if (pbv_src->type != GGML_TYPE_F32)
            pbv_src = ggml_cast(ctx, pbv_src, GGML_TYPE_F32);
        ggml_tensor* pbu = ggml_reshape_3d(ctx, pbu_src, head_dim, 1, n_heads);
        ggml_tensor* pbv = ggml_reshape_3d(ctx, pbv_src, head_dim, 1, n_heads);

        // q_with_bias_u = q + pos_bias_u: (hd, TT, H) + (hd, 1, H) broadcast
        ggml_tensor* q_u = ggml_add(ctx, q_t, pbu);
        ggml_tensor* q_v = ggml_add(ctx, q_t, pbv);

        // matrix_ac = q_with_bias_u @ k^T: (hd, TT, H) @ (hd, TT, H)^T → need per-head matmul
        // Use ggml_mul_mat which does a^T @ b over ne[0]:
        // k_t is (hd, TT, H). For q_u @ k^T per head: we need (TT, TT) per head
        // ggml_mul_mat(k_t, q_u): contracts ne[0]=hd → output (TT, TT, H). Correct!
        ggml_tensor* matrix_ac = ggml_mul_mat(ctx, k_t, q_u); // (TT, TT, H)

        // matrix_bd = q_with_bias_v @ p^T
        ggml_tensor* matrix_bd_raw = ggml_mul_mat(ctx, p, q_v); // (T_pos, TT, H)

        // Dump pre-shift matrix_bd
        if (std::strcmp(prefix + std::strlen(prefix) - 2, ".0") == 0 && std::strstr(prefix, "enc.")) {
            ggml_tensor* bd_raw_dump = ggml_scale(ctx, matrix_bd_raw, 1.0f);
            ggml_set_name(bd_raw_dump, "dump_enc0_bd_pre_shift");
            ggml_set_output(bd_raw_dump);
            ggml_build_forward_expand(gf, bd_raw_dump);
        }

        // Rel_shift: matrix_bd_raw ne=(2T-1, T, H) → (T, T, H)
        // ggml_reshape is a no-op on data (same flat bytes, different ne[]).
        // This matches Python's view() semantics exactly.
        ggml_tensor* matrix_bd = matrix_bd_raw;
        if ((int)matrix_bd->ne[0] != TT) {
            int T_pos = (int)matrix_bd->ne[0]; // 2T-1
            // Step 1: Pad zero on LEFT of ne[0] → (2T, T, H)
            ggml_tensor* zero_col = ggml_scale(
                ctx,
                ggml_cont(ctx, ggml_view_3d(ctx, matrix_bd, 1, TT, n_heads, matrix_bd->nb[1], matrix_bd->nb[2], 0)),
                0.0f);
            matrix_bd = ggml_concat(ctx, zero_col, matrix_bd, 0); // ne=(2T, T, H)

            // Step 2: Reshape (2T, T, H) → (T, 2T, H)
            matrix_bd = ggml_reshape_3d(ctx, matrix_bd, TT, 2 * TT, n_heads);

            // Step 3: Skip first row → ne[1] offset by 1 → (T, 2T-1, H)
            matrix_bd =
                ggml_view_3d(ctx, matrix_bd, TT, T_pos, n_heads, matrix_bd->nb[1], matrix_bd->nb[2], matrix_bd->nb[1]);
            matrix_bd = ggml_cont(ctx, matrix_bd);

            // Step 4: Reshape (T, 2T-1, H) → (2T-1, T, H) — view_as, NOT transpose
            matrix_bd = ggml_reshape_3d(ctx, matrix_bd, T_pos, TT, n_heads);

            // Step 5: Slice ne[0] to T → (T, T, H)
            if ((int)matrix_bd->ne[0] > TT) {
                matrix_bd = ggml_view_3d(ctx, matrix_bd, TT, TT, n_heads, matrix_bd->nb[1], matrix_bd->nb[2], 0);
                matrix_bd = ggml_cont(ctx, matrix_bd);
            }
        }

        // scores = (matrix_ac + matrix_bd) / sqrt(d_k)
        float scale = 1.0f / std::sqrt((float)head_dim);
        ggml_tensor* scores = ggml_add(ctx, matrix_ac, matrix_bd);
        scores = ggml_scale(ctx, scores, scale);

        // Dump score statistics for enc.0
        if (std::strcmp(prefix + std::strlen(prefix) - 2, ".0") == 0 && std::strstr(prefix, "enc.")) {
            ggml_tensor* ac_dump = ggml_scale(ctx, matrix_ac, 1.0f);
            ggml_set_name(ac_dump, "dump_enc0_matrix_ac");
            ggml_set_output(ac_dump);
            ggml_build_forward_expand(gf, ac_dump);
            ggml_tensor* bd_dump = ggml_scale(ctx, matrix_bd, 1.0f);
            ggml_set_name(bd_dump, "dump_enc0_matrix_bd");
            ggml_set_output(bd_dump);
            ggml_build_forward_expand(gf, bd_dump);
            ggml_tensor* sc_dump = ggml_scale(ctx, scores, 1.0f);
            ggml_set_name(sc_dump, "dump_enc0_scores_pre_softmax");
            ggml_set_output(sc_dump);
            ggml_build_forward_expand(gf, sc_dump);
        }

        // Softmax over ne[0] (key dim)
        scores = ggml_soft_max(ctx, scores); // (TT, TT, H)

        // Weighted sum: attn_out = scores @ v
        // v_t is (hd, TT, H). scores is (TT, TT, H).
        // For per-head: out[h] = scores[h] @ v[h] → (TT, hd) per head
        // ggml_mul_mat(v_t, scores): contracts ne[0]. v_t ne[0]=hd, scores ne[0]=TT. No match.
        // Need: scores^T @ v_t per head. But that's awkward.
        // Alternative: use the permuted layout. Let me think...
        // scores: (TT_key, TT_query, H) — attention weights
        // v_t: (hd, TT_key, H)
        // Want: out = sum_k scores[k,q,h] * v[d,k,h] for each (d,q,h)
        // This is: out = v @ scores^T per head → ggml_mul_mat(scores, v_t) with transpose
        // ggml_mul_mat(a, b) = a^T @ b over ne[0]
        // ggml_mul_mat(scores, v_t): a=scores (TT, TT, H), b=v_t (hd, TT, H)
        // contracts ne[0]: scores.ne[0]=TT, v_t.ne[0]=hd. No match!

        // Let me use a different approach: transpose scores to (TT_q, TT_k, H), then mul_mat with v_t
        ggml_tensor* scores_t = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3)); // (TT_q, TT_k, H)
        // Now ggml_mul_mat(v_t, scores_t): contracts ne[0]. v_t.ne[0]=hd, scores_t.ne[0]=TT_q. Still no.

        // Actually: we want out[d][q][h] = sum_k attn[k][q][h] * v[d][k][h]
        // This is: for each h: out[:,q] = V[:,:] @ attn[:,q]
        // In ggml: ggml_mul_mat(v_t_permuted, attn_col)
        // Let's just fall back to flash_attn_ext with the combined scores
        // Actually the simplest: convert to flash_attn compatible by pre-computing
        // the K with position info baked in. But that's not quite right.

        // Pragmatic approach: just do it with ggml_mul_mat by matching dims
        // v_t: (hd, TT, H), scores: (TT_k, TT_q, H)
        // Transpose v to (TT, hd, H) then mul_mat:
        ggml_tensor* v_p = ggml_cont(ctx, ggml_permute(ctx, v_t, 1, 0, 2, 3)); // (TT, hd, H)
        // ggml_mul_mat(v_p, scores): contracts ne[0]=TT. Output: (hd, TT_q, H). Correct!
        attn = ggml_mul_mat(ctx, v_p, scores); // (hd, TT, H)
        // Permute to head-concatenated layout: (hd, TT, H) → (hd, H, TT)
        // so reshape to (D, TT) gives [h0_d0..h0_d63, h1_d0..h7_d63] per time step,
        // matching Python's transpose(1,2).view(B, T, D) which concatenates heads.
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)); // (hd, H, TT)
        attn = ggml_reshape_2d(ctx, attn, D, TT);
    } else {
        // Fallback: simple scaled dot-product without relative position
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // (hd, TT, H)
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
        float scale = 1.0f / std::sqrt((float)head_dim);
        attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx, attn, D, TT);
    }

    // Mark raw attention output (before output proj)
    // Use ggml_scale(1.0) as identity op to create a graph node we can read
    if (std::strcmp(prefix + std::strlen(prefix) - 2, ".0") == 0 && std::strstr(prefix, "enc.")) {
        attn = ggml_scale(ctx, attn, 1.0f); // identity — creates a readable node
        ggml_set_name(attn, "dump_enc0_raw_attn");
        ggml_set_output(attn);
    }

    // Output projection
    ggml_tensor* attn_out = ggml_mul_mat(ctx, W("sa.lo.weight"), attn);
    ggml_tensor* lo_b = W("sa.lo.bias");
    if (lo_b)
        attn_out = ggml_add(ctx, attn_out, lo_b);

    if (std::strcmp(prefix + std::strlen(prefix) - 2, ".0") == 0 && std::strstr(prefix, "enc.")) {
        ggml_set_name(attn_out, "dump_enc0_attn_out");
        ggml_set_output(attn_out);
    }

    x = ggml_add(ctx, residual, attn_out);

    // ---- Feedforward with LayerNorm ----
    residual = x;
    ggml_tensor* nff_w = W("nff.weight");
    ggml_tensor* nff_b = W("nff.bias");
    xn = ggml_norm(ctx, x, 1e-5f);
    if (nff_w)
        xn = ggml_mul(ctx, xn, nff_w);
    if (nff_b)
        xn = ggml_add(ctx, xn, nff_b);

    // FFN: w_1 (512→2048) → SiLU → w_2 (2048→512)
    ggml_tensor* ff = ggml_mul_mat(ctx, W("ff.w_1.weight"), xn);
    ggml_tensor* ff_b1 = W("ff.w_1.bias");
    if (ff_b1)
        ff = ggml_add(ctx, ff, ff_b1);
    ff = ggml_silu(ctx, ff);
    ff = ggml_mul_mat(ctx, W("ff.w_2.weight"), ff);
    ggml_tensor* ff_b2 = W("ff.w_2.bias");
    if (ff_b2)
        ff = ggml_add(ctx, ff, ff_b2);

    x = ggml_add(ctx, residual, ff);
    return x;
}

// Build the full conformer encoder graph.
// Returns a ggml_cgraph* with "encoder_out" as the output tensor.
static ggml_cgraph* build_graph_conformer_encoder(chatterbox_s3gen_context* c, int n_tokens_total) {
    const int D = 512;
    const int H = 8;
    const int HD = 64;
    const int FF = 2048;
    const int Tin = n_tokens_total;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

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
        if (lin_b)
            x = ggml_add(ctx0, x, lin_b);
    }
    // LayerNorm (embed.out.1)
    ggml_tensor* ln_w = T(c, "s3.fe.embed.out.1.weight");
    ggml_tensor* ln_b = T(c, "s3.fe.embed.out.1.bias");
    if (ln_w) {
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, ln_w);
        if (ln_b)
            x = ggml_add(ctx0, x, ln_b);
    }

    // Scale by sqrt(D) — matches EspnetRelPositionalEncoding.xscale
    x = ggml_scale(ctx0, x, std::sqrt((float)D));

    ggml_set_name(x, "dump_after_embed");
    ggml_set_output(x);

    // Pre-lookahead conv (causal): conv1(k=4, pad_left=3) + LeakyReLU + conv2(k=3, pad_left=2) + residual
    {
        ggml_tensor* pla_c1_w = T(c, "s3.fe.pla.conv1.weight");
        ggml_tensor* pla_c1_b = T(c, "s3.fe.pla.conv1.bias");
        ggml_tensor* pla_c2_w = T(c, "s3.fe.pla.conv2.weight");
        ggml_tensor* pla_c2_b = T(c, "s3.fe.pla.conv2.bias");
        if (pla_c1_w && pla_c2_w) {
            ggml_tensor* residual = x;
            // x is (D=512, Tin) from embedding. Conv1d expects (T, C_in) → transpose.
            x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // now (Tin, D=512)

            // Helper: pad one side (left or right) with zeros from x via scale(0),
            // then conv with pad=0
            auto padded_conv1d = [&](ggml_tensor* input, ggml_tensor* w, ggml_tensor* b, int pad_size,
                                     bool pad_right) -> ggml_tensor* {
                ggml_tensor* pad_view =
                    ggml_cont(ctx0, ggml_view_2d(ctx0, input, pad_size, (int)input->ne[1], input->nb[1], 0));
                ggml_tensor* zeros = ggml_scale(ctx0, pad_view, 0.0f);
                ggml_tensor* padded;
                if (pad_right) {
                    padded = ggml_concat(ctx0, input, zeros, 0); // [x, zeros]
                } else {
                    padded = ggml_concat(ctx0, zeros, input, 0); // [zeros, x]
                }
                ggml_tensor* out = ggml_conv_1d(ctx0, w, padded, 1, 0, 1);
                if (b)
                    out = ggml_add(ctx0, out, ggml_reshape_2d(ctx0, b, 1, (int)b->ne[0]));
                return out;
            };

            // Conv1(k=4, RIGHT-pad 3): lookahead conv — looks 3 steps into the future
            x = padded_conv1d(x, pla_c1_w, pla_c1_b, 3, /*pad_right=*/true);
            ggml_set_name(x, "dump_pla_conv1");
            ggml_set_output(x);
            x = ggml_leaky_relu(ctx0, x, 0.01f, false); // Python default slope=0.01

            // Conv2(k=3, LEFT-pad 2): causal conv
            x = padded_conv1d(x, pla_c2_w, pla_c2_b, 2, /*pad_right=*/false);
            // Transpose back to (D, Tin) to match residual layout
            x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (D, Tin)
            // Residual
            x = ggml_add(ctx0, x, residual);
        }
    }

    // Mark PLA output for dump
    ggml_set_name(x, "dump_after_pla");
    ggml_set_output(x);

    // Generate Espnet-style relative positional encoding: sinusoidal, shape (D, 2*T-1)
    // pe[i] = sin(pos * freq) for even dims, cos(pos * freq) for odd dims
    // pos ranges from -(T-1) to +(T-1), freq = 1/10000^(2i/D)
    int T_pos_pre = 2 * Tin - 1;
    ggml_tensor* pos_emb_pre = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_pos_pre);
    ggml_set_name(pos_emb_pre, "pos_emb_pre");
    ggml_set_input(pos_emb_pre);

    // 6 conformer blocks (pre-upsample)
    for (int i = 0; i < 6; i++) {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "s3.fe.enc.%d", i);
        x = build_conformer_block(ctx0, gf, c, x, Tin, prefix, H, HD, D, FF, pos_emb_pre);
        char dname[32];
        std::snprintf(dname, sizeof(dname), "dump_enc_%d", i);
        ggml_set_name(x, dname);
        ggml_set_output(x);
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

    // Upsample1D conv: left-pad(4) → Conv1d(512, 512, k=5, s=1, pad=0)
    // Python: F.pad(x, (stride*2, 0)) → conv → output same length as input
    {
        ggml_tensor* ul_w = T(c, "s3.fe.ul.conv.weight");
        ggml_tensor* ul_b = T(c, "s3.fe.ul.conv.bias");
        if (ul_w) {
            // x is (D, T2). Conv1d expects (T, C) → transpose
            ggml_tensor* xt = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (T2, D)
            // Left-pad by 4 zeros
            ggml_tensor* pad_view = ggml_cont(ctx0, ggml_view_2d(ctx0, xt, 4, (int)xt->ne[1], xt->nb[1], 0));
            ggml_tensor* zeros = ggml_scale(ctx0, pad_view, 0.0f);
            ggml_tensor* padded = ggml_concat(ctx0, zeros, xt, 0); // (T2+4, D)
            // Conv1d(k=5, s=1, pad=0): output length = T2+4-5+1 = T2
            ggml_tensor* conv_out = ggml_conv_1d(ctx0, ul_w, padded, 1, 0, 1);
            if (ul_b)
                conv_out = ggml_add(ctx0, conv_out, ggml_reshape_2d(ctx0, ul_b, 1, (int)ul_b->ne[0]));
            // conv_out is (T2, D) — transpose back to (D, T2)
            x = ggml_cont(ctx0, ggml_transpose(ctx0, conv_out)); // (D, T2)
        }
    }

    // Re-embed: up_embed.out.0 (Linear) + up_embed.out.1 (LayerNorm)
    ggml_tensor* uemb_w = T(c, "s3.fe.uemb.out.0.weight");
    ggml_tensor* uemb_b = T(c, "s3.fe.uemb.out.0.bias");
    if (uemb_w) {
        x = ggml_mul_mat(ctx0, uemb_w, x);
        if (uemb_b)
            x = ggml_add(ctx0, x, uemb_b);
    }
    ggml_tensor* uln_w = T(c, "s3.fe.uemb.out.1.weight");
    ggml_tensor* uln_b = T(c, "s3.fe.uemb.out.1.bias");
    if (uln_w) {
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, uln_w);
        if (uln_b)
            x = ggml_add(ctx0, x, uln_b);
    }

    // xscale — same as pre-embed path; Python's up_embed uses RelPositionalEncoding
    // which multiplies x * sqrt(d_model) inside its forward()
    x = ggml_scale(ctx0, x, std::sqrt((float)D));

    // Generate pos encoding for upsampled sequence
    int T_pos_post = 2 * T2 - 1;
    ggml_tensor* pos_emb_post = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_pos_post);
    ggml_set_name(pos_emb_post, "pos_emb_post");
    ggml_set_input(pos_emb_post);

    // 4 conformer blocks (post-upsample)
    for (int i = 0; i < 4; i++) {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "s3.fe.ue.%d", i);
        x = build_conformer_block(ctx0, gf, c, x, T2, prefix, H, HD, D, FF, pos_emb_post);
    }

    // Final LayerNorm
    ggml_tensor* an_w = T(c, "s3.fe.an.weight");
    ggml_tensor* an_b = T(c, "s3.fe.an.bias");
    if (an_w) {
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, an_w);
        if (an_b)
            x = ggml_add(ctx0, x, an_b);
    }

    // Project to 80D: encoder_proj (80, 512)
    ggml_tensor* proj_w = TR(c, "s3.flow.encoder_proj.weight");
    ggml_tensor* proj_b = T(c, "s3.flow.encoder_proj.bias");
    x = ggml_mul_mat(ctx0, proj_w, x);
    if (proj_b)
        x = ggml_add(ctx0, x, proj_b);

    ggml_set_name(x, "encoder_out");
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);
    return gf;
}

// Run the conformer encoder via ggml graph.
// Returns (80, T_mel) channel-first mel-space encoder output.
static std::vector<float> run_conformer_encoder(chatterbox_s3gen_context* c, const int32_t* speech_tokens, int n_tokens,
                                                const int32_t* prompt_tokens, int n_prompt) {
    const int total = n_prompt + n_tokens;
    const int T_mel = total * 2; // 2x upsample

    // Build token ID array: [prompt | speech]
    std::vector<int32_t> all_tokens(total);
    if (n_prompt > 0)
        std::memcpy(all_tokens.data(), prompt_tokens, n_prompt * sizeof(int32_t));
    std::memcpy(all_tokens.data() + n_prompt, speech_tokens, n_tokens * sizeof(int32_t));

    // Build and run graph
    ggml_cgraph* gf = build_graph_conformer_encoder(c, total);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "s3gen: failed to alloc conformer graph\n");
        return {};
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "token_ids"), all_tokens.data(), 0, total * sizeof(int32_t));

    // Fill sinusoidal relative positional encodings (EspnetRelPositionalEncoding).
    // Python builds pe_positive[0..T-1] = sin/cos(pos*freq), flips it, then
    // appends pe_negative[1..T-1]. Result: index 0 = position +(T-1) (most
    // positive), index T-1 = position 0, index 2T-2 = position -(T-1).
    // In ggml layout: (D, 2T-1) with ne[0]=D, ne[1]=2T-1
    auto fill_pos_enc = [&](const char* name, int T) {
        ggml_tensor* pe_t = ggml_graph_get_tensor(gf, name);
        if (!pe_t)
            return;
        int T_pos = 2 * T - 1;
        const int D = 512;
        std::vector<float> pe_data((size_t)D * T_pos);
        for (int p = 0; p < T_pos; p++) {
            float pos = (float)((T - 1) - p); // +(T-1) to -(T-1), matching Python
            for (int i = 0; i < D / 2; i++) {
                float freq = 1.0f / std::pow(10000.0f, (float)(2 * i) / (float)D);
                pe_data[p * D + 2 * i] = std::sin(pos * freq);
                pe_data[p * D + 2 * i + 1] = std::cos(pos * freq);
            }
        }
        ggml_backend_tensor_set(pe_t, pe_data.data(), 0, pe_data.size() * sizeof(float));
    };
    fill_pos_enc("pos_emb_pre", total);
    fill_pos_enc("pos_emb_post", T_mel);

    // Two-pass execution: first compute to get matrix_bd_raw, then apply
    // CPU-side rel_shift and re-run for the attention + FFN.
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "s3gen: conformer compute failed\n");
        return {};
    }

    // Single-pass: rel_shift is now computed in the ggml graph via
    // pad+reshape+view+permute+slice (matching Python's view semantics).

    // Dump per-layer RMS if CRISPASR_S3GEN_DUMP=1
    {
        const char* dump_env = std::getenv("CRISPASR_S3GEN_DUMP");
        if (dump_env && dump_env[0] == '1') {
            auto dump_rms = [&](const char* name) {
                ggml_tensor* t = ggml_graph_get_tensor(gf, name);
                if (!t)
                    return;
                size_t n = ggml_nelements(t);
                std::vector<float> d(n);
                ggml_backend_tensor_get(t, d.data(), 0, n * sizeof(float));
                float rms = 0;
                for (auto v : d)
                    rms += v * v;
                rms = std::sqrt(rms / n);
                fprintf(stderr, "s3gen[enc]: %-16s ne=(%lld,%lld) rms=%.6f\n", name, (long long)t->ne[0],
                        (long long)t->ne[1], rms);
            };
            dump_rms("dump_after_embed");
            dump_rms("dump_pla_conv1");
            dump_rms("dump_after_pla");
            dump_rms("dump_enc0_pre_attn_norm");
            dump_rms("dump_enc0_linear_pos_out");
            dump_rms("dump_enc0_bd_pre_shift");
            dump_rms("dump_enc0_matrix_ac");
            dump_rms("dump_enc0_bd_pre_shift");
            dump_rms("dump_enc0_matrix_bd");
            // Element-level dump: head 0, first 5x5 of matrix_bd for diff comparison
            // matrix_bd: ne[0]=TT, ne[1]=TT, ne[2]=H. Element (k, q, h) at data[h*TT*TT + q*TT + k]
            {
                ggml_tensor* bd_t = ggml_graph_get_tensor(gf, "dump_enc0_matrix_bd");
                if (bd_t) {
                    int ne0 = (int)bd_t->ne[0];
                    int ne1 = (int)bd_t->ne[1];
                    int ne2 = (bd_t->ne[2] > 1) ? (int)bd_t->ne[2] : 1;
                    size_t n = ggml_nelements(bd_t);
                    std::vector<float> bd(n);
                    ggml_backend_tensor_get(bd_t, bd.data(), 0, n * sizeof(float));
                    fprintf(stderr, "s3gen[enc]: matrix_bd ne=(%d,%d,%d) total=%zu\n", ne0, ne1, ne2, n);
                    // For head 0: data layout depends on ne ordering
                    // ne[0]=key, ne[1]=query, ne[2]=head. Element (k,q,h) at h*ne1*ne0 + q*ne0 + k
                    // Also dump pre-shift bd
                    {
                        ggml_tensor* bdr = ggml_graph_get_tensor(gf, "dump_enc0_bd_pre_shift");
                        if (bdr) {
                            int ne0r = (int)bdr->ne[0], ne1r = (int)bdr->ne[1];
                            size_t nr = ggml_nelements(bdr);
                            std::vector<float> bdr_d(nr);
                            ggml_backend_tensor_get(bdr, bdr_d.data(), 0, nr * sizeof(float));
                            fprintf(stderr, "s3gen[enc]: bd_pre_shift ne=(%d,%d,%lld) first 10: ", ne0r, ne1r,
                                    (long long)bdr->ne[2]);
                            for (int i = 0; i < 10 && i < (int)nr; i++)
                                fprintf(stderr, "%.2f ", bdr_d[i]);
                            fprintf(stderr, "\n");
                        }
                    }
                    // Print first 5 values and a 5x5 block
                    fprintf(stderr, "s3gen[enc]: matrix_bd first 10 raw: ");
                    for (int i = 0; i < 10 && i < (int)n; i++)
                        fprintf(stderr, "%.2f ", bd[i]);
                    fprintf(stderr, "\n");
                    fprintf(stderr, "s3gen[enc]: matrix_bd h0 [q=0:5, k=0:5]:\n");
                    for (int q = 0; q < 5; q++) {
                        fprintf(stderr, "s3gen[enc]:  ");
                        for (int k = 0; k < 5; k++)
                            fprintf(stderr, " %8.2f", bd[q * ne0 + k]);
                        fprintf(stderr, "\n");
                    }
                    // Python reference h0 [0:5,0:5]:
                    fprintf(stderr, "s3gen[enc]: Python ref h0 [0:5,0:5]:\n");
                    fprintf(stderr, "s3gen[enc]:   [24.70, 23.75, 20.29, 15.85, 12.23]\n");
                    fprintf(stderr, "s3gen[enc]:   [32.29, 50.34, 60.43, 60.74, 54.70]\n");
                    fprintf(stderr, "s3gen[enc]:   [18.26, 38.67, 56.16, 64.84, 64.05]\n");
                    fprintf(stderr, "s3gen[enc]:   [-1.14, 16.49, 39.36, 59.26, 69.95]\n");
                    fprintf(stderr, "s3gen[enc]:   [-9.08, -1.82, 16.95, 41.58, 63.07]\n");
                }
            }
            dump_rms("dump_enc0_scores_pre_softmax");
            dump_rms("dump_enc0_raw_attn");
            dump_rms("dump_enc0_attn_out");
            for (int i = 0; i < 6; i++) {
                char dn[32];
                std::snprintf(dn, sizeof(dn), "dump_enc_%d", i);
                dump_rms(dn);
            }
        }
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
static ggml_tensor* causal_conv1d(ggml_context* ctx, ggml_tensor* x, // input: (T, C_in) in ggml layout
                                  ggml_tensor* weight,               // conv weight: (K, C_in, C_out)
                                  ggml_tensor* bias                  // (C_out) or nullptr
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
static ggml_tensor* causal_block1d(ggml_context* ctx, ggml_tensor* x, // (C, T)
                                   ggml_tensor* conv_w, ggml_tensor* conv_b, ggml_tensor* ln_w, ggml_tensor* ln_b) {
    x = causal_conv1d(ctx, x, conv_w, conv_b);
    // LayerNorm over channel dim: transpose (C, T) → (T, C), norm, transpose back
    // After conv1d: x is (T, C) in ggml layout
    // LayerNorm over C dimension (ne[0]=T, ne[1]=C → norm over C)
    // ggml_norm normalizes over ne[0] by default. We need to normalize over C (ne[1]).
    // Transpose to (C, T), norm over C (now ne[0]), transpose back.
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (C, T)
    x = ggml_norm(ctx, x, 1e-5f);
    if (ln_w)
        x = ggml_mul(ctx, x, ln_w);
    if (ln_b)
        x = ggml_add(ctx, x, ln_b);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C)
    // SiLU ≈ Mish
    x = ggml_silu(ctx, x);
    return x;
}

// Helper: CausalResnetBlock1D — block1 + time_mlp + block2 + residual
static ggml_tensor* causal_resnet_block(ggml_context* ctx, ggml_tensor* x, ggml_tensor* t_emb,
                                        chatterbox_s3gen_context* c, const char* prefix, ggml_tensor* mask) {
    char key[64];
    auto W = [&](const char* suffix) -> ggml_tensor* {
        std::snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
        return core_gguf::try_get(c->tensors, key);
    };

    ggml_tensor* residual = x;

    // block1: CausalConv1d(k=3) + LayerNorm + Mish
    x = causal_block1d(ctx, x, W("b1.0.weight"), W("b1.0.bias"), W("b1.2.weight"), W("b1.2.bias"));
    // DISABLED: if (mask) x = ggml_mul(ctx, x, mask);

    // Time MLP: linear(1024 → C_out) → add broadcast over T
    // t_emb is (1024,), W is (C_out, 1024) → t_proj is (C_out,)
    ggml_tensor* t_proj = ggml_mul_mat(ctx, W("mlp.1.weight"), t_emb);
    ggml_tensor* t_b = W("mlp.1.bias");
    if (t_b)
        t_proj = ggml_add(ctx, t_proj, t_b);
    // x is (T, C_out) after causal_block1d. t_proj is (C_out,).
    // Reshape t_proj to (1, C_out) so it broadcasts over T (ne[0]).
    t_proj = ggml_reshape_2d(ctx, t_proj, 1, (int)t_proj->ne[0]);
    // Now (T, C_out) + (1, C_out) → broadcasts T times
    x = ggml_add(ctx, x, t_proj);

    // block2: CausalConv1d(k=3) + LayerNorm + Mish
    x = causal_block1d(ctx, x, W("b2.0.weight"), W("b2.0.bias"), W("b2.2.weight"), W("b2.2.bias"));
    // DISABLED: if (mask) x = ggml_mul(ctx, x, mask);

    // Residual conv (if dimensions differ)
    ggml_tensor* rc_w = W("rc.weight");
    if (rc_w) {
        ggml_tensor* rc_b = W("rc.bias");
        residual = ggml_conv_1d(ctx, rc_w, residual, 1, 0, 1);
        if (rc_b)
            residual = ggml_add(ctx, residual, ggml_reshape_2d(ctx, rc_b, 1, (int)rc_b->ne[0]));
    }

    return ggml_add(ctx, x, residual);
}

// Helper: BasicTransformerBlock — norm1 → self-attn → norm3 → FF(GEGLU)
static ggml_tensor* basic_transformer_block(ggml_context* ctx, ggml_tensor* x, // (C, T) channels-first
                                            chatterbox_s3gen_context* c, const char* prefix,
                                            ggml_tensor* attn_mask, // causal mask or nullptr
                                            int n_heads, int head_dim) {
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
    if (n1w)
        xn = ggml_mul(ctx, xn, n1w);
    if (n1b)
        xn = ggml_add(ctx, xn, n1b);

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
    if (o_b)
        attn = ggml_add(ctx, attn, o_b);

    xt = ggml_add(ctx, residual, attn);

    // norm3 → FF (GEGLU pattern: up_proj with 2x output, split, gate)
    residual = xt;
    xn = ggml_norm(ctx, xt, 1e-5f);
    ggml_tensor* n3w = W("norm3.weight");
    ggml_tensor* n3b = W("norm3.bias");
    if (n3w)
        xn = ggml_mul(ctx, xn, n3w);
    if (n3b)
        xn = ggml_add(ctx, xn, n3b);

    // FF up: Linear(C → 4C) + GELU (not GEGLU — decoder uses act_fn="gelu")
    ggml_tensor* ff_up = ggml_mul_mat(ctx, W("ff.up.weight"), xn);
    ggml_tensor* ff_up_b = W("ff.up.bias");
    if (ff_up_b)
        ff_up = ggml_add(ctx, ff_up, ff_up_b);
    ff_up = ggml_gelu(ctx, ff_up);

    // FF down: Linear(4C → C)
    ggml_tensor* ff_out = ggml_mul_mat(ctx, W("ff.down.weight"), ff_up);
    ggml_tensor* ff_down_b = W("ff.down.bias");
    if (ff_down_b)
        ff_out = ggml_add(ctx, ff_out, ff_down_b);

    xt = ggml_add(ctx, residual, ff_out);

    // Transpose back to (C, T)
    return ggml_cont(ctx, ggml_transpose(ctx, xt));
}

// Build the full UNet1D denoiser graph for one CFM step.
// x_in: (320, T) = concat[x(80), mu(80), spks(80), cond(80)]
// t_emb: (1024,) time embedding
// Returns: (80, T) velocity prediction
static ggml_cgraph* build_graph_unet1d(chatterbox_s3gen_context* c, int T_mel) {
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
        if (ds_w)
            x = causal_conv1d(ctx0, x, ds_w, ds_b);
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
                hidden = ggml_view_2d(ctx0, hidden, T_x, (int)hidden->ne[1], hidden->nb[1], 0);
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
        if (us_w)
            x = causal_conv1d(ctx0, x, us_w, us_b);
    }

    // ---- Final block + projection ----
    {
        ggml_tensor* fb_w = T(c, "s3.fd.fb.block.0.weight");
        ggml_tensor* fb_b = T(c, "s3.fd.fb.block.0.bias");
        ggml_tensor* fb_ln_w = T(c, "s3.fd.fb.block.2.weight");
        ggml_tensor* fb_ln_b = T(c, "s3.fd.fb.block.2.bias");
        if (fb_w)
            x = causal_block1d(ctx0, x, fb_w, fb_b, fb_ln_w, fb_ln_b);
        if (mask)
            x = ggml_mul(ctx0, x, mask);

        // Final projection: Conv1d(256→80, k=1)
        ggml_tensor* fp_w = T(c, "s3.fd.fp.weight");
        ggml_tensor* fp_b = T(c, "s3.fd.fp.bias");
        if (fp_w) {
            x = ggml_conv_1d(ctx0, fp_w, x, 1, 0, 1);
            if (fp_b)
                x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, fp_b, 1, (int)fp_b->ne[0]));
        }
    }

    if (mask)
        x = ggml_mul(ctx0, x, mask);
    ggml_set_name(x, "denoiser_out");
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);
    return gf;
}

// ── CFM Euler solver with UNet1D denoiser ───────────────────────

static std::vector<float> cfm_euler_solve(chatterbox_s3gen_context* c,
                                          const std::vector<float>& mu,      // (80, T) encoder output (channel-first)
                                          const std::vector<float>& cond,    // (80, T) conditioning mel (channel-first)
                                          const std::vector<float>& spk_emb, // (80,) projected speaker embedding
                                          int T_mel, int n_steps, float cfg_rate, bool meanflow = false,
                                          bool dump_stages = false) {
    // Generate time schedule
    std::vector<float> t_span(n_steps + 1);
    for (int i = 0; i <= n_steps; i++) {
        float t = (float)i / (float)n_steps;
        if (!meanflow) {
            t_span[i] = 1.0f - std::cos(t * 0.5f * (float)M_PI); // cosine schedule
        } else {
            t_span[i] = t; // linear schedule for meanflow
        }
    }

    // Start from noise
    std::vector<float> x(80 * T_mel);
    uint64_t rng = 42;
    for (size_t i = 0; i < x.size(); i++) {
        float u1 = (float)((rng = rng * 6364136223846793005ULL + 1) >> 33) / (float)(1ULL << 31);
        float u2 = (float)((rng = rng * 6364136223846793005ULL + 1) >> 33) / (float)(1ULL << 31);
        if (u1 < 1e-7f)
            u1 = 1e-7f;
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

        // Build conditioned input: [x, mu, spks, cond] concatenated along channels
        // ggml tensor ne=(T_mel, 320): data stored as data[ch * T_mel + t]
        // (channel-first, T is fast axis)
        std::vector<float> unet_input(T_mel * 320, 0.0f);
        for (int ch = 0; ch < 80; ch++) {
            for (int t = 0; t < T_mel; t++) {
                unet_input[(ch + 0) * T_mel + t] = x[ch * T_mel + t];
                unet_input[(ch + 80) * T_mel + t] = mu[ch * T_mel + t];
                unet_input[(ch + 160) * T_mel + t] = spk_emb[ch]; // broadcast over T
                unet_input[(ch + 240) * T_mel + t] = cond[ch * T_mel + t];
            }
        }

        // Build unconditioned input for CFG: [x, 0, 0, 0]
        std::vector<float> unet_uncond(T_mel * 320, 0.0f);
        if (cfg_rate > 0.0f) {
            for (int ch = 0; ch < 80; ch++) {
                for (int t = 0; t < T_mel; t++) {
                    unet_uncond[ch * T_mel + t] = x[ch * T_mel + t];
                }
            }
        }

        // Time embedding: sinusoidal(320) → MLP(320→1024→1024)
        // For meanflow: also compute r embedding and mix via time_embed_mixer
        std::vector<float> t_sin = sinusoidal_embedding(t_val, 320);
        // Helper: run time MLP (shared for t and r)
        auto time_mlp = [&](std::vector<float>& emb) {
            ggml_tensor* tm1_w = T(c, "s3.fd.tm.linear_1.weight");
            ggml_tensor* tm1_b = T(c, "s3.fd.tm.linear_1.bias");
            ggml_tensor* tm2_w = T(c, "s3.fd.tm.linear_2.weight");
            ggml_tensor* tm2_b = T(c, "s3.fd.tm.linear_2.bias");
            if (!tm1_w || !tm2_w)
                return;
            std::vector<float> w1(1024 * 320), b1(1024, 0.0f);
            std::vector<float> w2(1024 * 1024), b2(1024, 0.0f);
            tensor_get_f32(tm1_w, w1.data(), 0, w1.size());
            if (tm1_b)
                tensor_get_f32(tm1_b, b1.data(), 0, b1.size());
            tensor_get_f32(tm2_w, w2.data(), 0, w2.size());
            if (tm2_b)
                tensor_get_f32(tm2_b, b2.data(), 0, b2.size());
            std::vector<float> h1(1024);
            for (int i = 0; i < 1024; i++) {
                float sum = b1[i];
                for (int j = 0; j < 320; j++)
                    sum += w1[i * 320 + j] * emb[j];
                h1[i] = sum;
            }
            // SiLU: x * sigmoid(x)
            for (int i = 0; i < 1024; i++) {
                float sig = 1.0f / (1.0f + std::exp(-h1[i]));
                h1[i] = h1[i] * sig;
            }
            emb.resize(1024);
            for (int i = 0; i < 1024; i++) {
                float sum = b2[i];
                for (int j = 0; j < 1024; j++)
                    sum += w2[i * 1024 + j] * h1[j];
                emb[i] = sum;
            }
        };
        time_mlp(t_sin); // t_sin is now (1024,) t embedding

        if (meanflow) {
            // Meanflow: compute r embedding, concat with t, mix
            std::vector<float> r_emb = sinusoidal_embedding(r_val, 320);
            time_mlp(r_emb); // r_emb is now (1024,) r embedding
            // Concat [t_emb, r_emb] → (2048,)
            std::vector<float> concat(2048);
            std::memcpy(concat.data(), t_sin.data(), 1024 * sizeof(float));
            std::memcpy(concat.data() + 1024, r_emb.data(), 1024 * sizeof(float));
            // time_embed_mixer: linear(2048 → 1024)
            ggml_tensor* tmx_w = T(c, "s3.fd.tmx.weight");
            if (tmx_w) {
                std::vector<float> w(1024 * 2048);
                tensor_get_f32(tmx_w, w.data(), 0, w.size());
                for (int i = 0; i < 1024; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < 2048; j++)
                        sum += w[i * 2048 + j] * concat[j];
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
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "unet_input"), input.data(), 0,
                                    input.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "time_emb"), t_sin.data(), 0,
                                    t_sin.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"), mask_data.data(), 0,
                                    mask_data.size() * sizeof(float));
            if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "s3gen: UNet1D compute failed\n");
                return {};
            }
            ggml_tensor* out = ggml_graph_get_tensor(gf, "denoiser_out");
            if (step == 0 && c->verbosity >= 1) {
                fprintf(stderr, "s3gen: denoiser out ne=(%d, %d)\n", (int)out->ne[0], (int)out->ne[1]);
            }
            size_t nb = ggml_nbytes(out);
            std::vector<float> v(nb / sizeof(float));
            ggml_backend_tensor_get(out, v.data(), 0, nb);
            return v;
        };

        // Run conditioned pass
        std::vector<float> v_cond = run_denoiser(unet_input);
        if (v_cond.empty())
            break;

        // Run unconditioned pass for CFG (skip for meanflow — distilled, no CFG)
        std::vector<float> v_uncond;
        if (cfg_rate > 0.0f && !meanflow) {
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
        // Denoiser output is ggml (ne[0]=T, ne[1]=C) → data[ch * T + t] (channel-first)
        for (int ch = 0; ch < use_C; ch++) {
            for (int t = 0; t < use_T; t++) {
                float vc = v_cond[ch * out_T + t];
                float v;
                if (cfg_rate > 0.0f && !v_uncond.empty()) {
                    float vu = v_uncond[ch * out_T + t];
                    v = (1.0f + cfg_rate) * vc - cfg_rate * vu;
                } else {
                    v = vc;
                }
                x[ch * T_mel + t] += dt * v;
            }
        }

        if (c->verbosity >= 2 || dump_stages) {
            float rms = 0.0f;
            for (auto v : x)
                rms += v * v;
            rms = std::sqrt(rms / x.size());
            fprintf(stderr, "s3gen: CFM step %d/%d (t=%.3f→%.3f) x_rms=%.4f\n", step + 1, n_steps, t_val, r_val, rms);
        }
        if (dump_stages) {
            // Print first 5 values at (t=0, ch=0..4) for diff comparison
            fprintf(stderr, "s3gen[dump]: cfm_step_%d (t=0,ch=0..4): ", step);
            for (int ch = 0; ch < 5; ch++)
                fprintf(stderr, "%.6f ", x[ch * T_mel + 0]);
            fprintf(stderr, "\n");
            // Also velocity rms
            float v_rms = 0;
            for (auto v : v_cond)
                v_rms += v * v;
            v_rms = std::sqrt(v_rms / v_cond.size());
            fprintf(stderr, "s3gen[dump]: velocity_rms=%.4f\n", v_rms);
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
static std::vector<float> run_f0_predictor(chatterbox_s3gen_context* c,
                                           const std::vector<float>& mel, // (80, T_mel) channel-first
                                           int T_mel) {
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
        if (!wt)
            continue;

        int C_in = (layer == 0) ? 80 : C;
        int C_out = C;

        // Read weights with automatic dequantization (handles F32/F16/Q8_0/Q4_K/etc)
        size_t n_elem = (size_t)K * C_in * C_out;
        std::vector<float> w_f32(n_elem);
        std::vector<float> b(C_out, 0.0f);
        {
            std::vector<char> raw(ggml_nbytes(wt));
            ggml_backend_tensor_get(wt, raw.data(), 0, raw.size());
            if (wt->type == GGML_TYPE_F32) {
                std::memcpy(w_f32.data(), raw.data(), n_elem * sizeof(float));
            } else {
                // Use ggml's built-in dequantize for any quantized type
                const auto* type_traits = ggml_get_type_traits(wt->type);
                if (type_traits && type_traits->to_float) {
                    type_traits->to_float(raw.data(), w_f32.data(), (int)n_elem);
                } else if (wt->type == GGML_TYPE_F16) {
                    const ggml_fp16_t* w16 = (const ggml_fp16_t*)raw.data();
                    for (size_t i = 0; i < n_elem; i++)
                        w_f32[i] = ggml_fp16_to_fp32(w16[i]);
                }
            }
        }
        if (bt)
            ggml_backend_tensor_get(bt, b.data(), 0, C_out * sizeof(float));

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
                    if (tt < 0 || tt >= T_mel)
                        continue;
                    for (int ci = 0; ci < C_in; ci++) {
                        // w layout: (K, C_in, C_out) → w[k * C_in * C_out + ci * C_out + co]
                        // Actually ggml stores as ne[0]=K, ne[1]=C_in, ne[2]=C_out
                        // Memory: w[co * C_in * K + ci * K + k]
                        sum += w_f32[co * C_in * K + ci * K + k] * x[tt * C_in + ci];
                    }
                }
                // ELU activation
                if (sum < 0)
                    sum = std::exp(sum) - 1.0f;
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
        {
            std::vector<char> raw(ggml_nbytes(cls_w));
            ggml_backend_tensor_get(cls_w, raw.data(), 0, raw.size());
            if (cls_w->type == GGML_TYPE_F32) {
                std::memcpy(cw_f32.data(), raw.data(), C * sizeof(float));
            } else {
                const auto* type_traits = ggml_get_type_traits(cls_w->type);
                if (type_traits && type_traits->to_float) {
                    type_traits->to_float(raw.data(), cw_f32.data(), C);
                } else if (cls_w->type == GGML_TYPE_F16) {
                    const ggml_fp16_t* cw16 = (const ggml_fp16_t*)raw.data();
                    for (int i = 0; i < C; i++)
                        cw_f32[i] = ggml_fp16_to_fp32(cw16[i]);
                }
            }
        }
        float cb = 0.0f;
        if (cls_b)
            ggml_backend_tensor_get(cls_b, &cb, 0, sizeof(float));

        for (int t = 0; t < T_mel; t++) {
            float sum = cb;
            for (int i = 0; i < C; i++)
                sum += cw_f32[i] * x[t * C + i];
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

// stage_dump: if non-null, map is filled with named stage outputs after graph compute.
static std::vector<float> hift_vocoder_cpu(chatterbox_s3gen_context* c,
                                           const std::vector<float>& mel, // (80, T_mel) channel-first
                                           int T_mel, const float* source_stft_cf = nullptr, int T_src_ext = 0,
                                           std::map<std::string, std::vector<float>>* stage_dump = nullptr) {
    if (c->verbosity >= 1) {
        float mel_rms = 0, mel_max = 0;
        for (size_t i = 0; i < mel.size(); i++) {
            mel_rms += mel[i] * mel[i];
            if (std::abs(mel[i]) > mel_max)
                mel_max = std::abs(mel[i]);
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
        if (cpre_b)
            x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, cpre_b, 1, (int)cpre_b->ne[0]));
    }
    ggml_set_name(x, "voc_conv_pre");
    ggml_set_output(x);

    // Source STFT input (from SineGen → STFT): 18 channels (real + imag)
    // Total audio length = T_mel * 120 (upsample) * 4 (iSTFT hop)
    // Source operates at audio rate: T_audio = T_mel * upsample_total * istft_hop
    // STFT of source: n_frames ≈ T_audio
    // For simplicity, we compute source length from T_mel and provide it as input
    int T_audio = T_mel * 120 * 4; // total audio samples
    // Upstream torch.stft runs with center=True and pad_mode='reflect',
    // which yields floor(T_audio / hop) + 1 frames.
    int T_src = T_audio / istft_hop + 1;
    if (source_stft_cf && T_src_ext > 0)
        T_src = T_src_ext;
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
            if (up_b)
                x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, up_b, 1, (int)up_b->ne[0]));
        }
        // Reflection pad at last upsample stage: ReflectionPad1d((1, 0))
        // Python: if i == num_upsamples - 1: x = self.reflection_pad(x)
        if (stage == 2) {
            // Reflect-pad 1 sample on left: x[-1] is prepended
            // x has ne=(T, C). Take the second sample (index 1), prepend it.
            int T_x = (int)x->ne[0];
            int C_x = (int)x->ne[1];
            // reflection pad left=1: new[0] = x[1], new[1..T] = x[0..T-1]
            ggml_tensor* pad_sample = ggml_view_2d(ctx0, x, 1, C_x, x->nb[1], 1 * x->nb[0]); // x[:,1]
            pad_sample = ggml_cont(ctx0, pad_sample);
            x = ggml_concat(ctx0, pad_sample, x, 0); // prepend → (T+1, C)
        }
        {
            char uname[32];
            std::snprintf(uname, sizeof(uname), "voc_ups_%d", stage);
            ggml_set_name(x, uname);
            ggml_set_output(x);
        }

        // Source fusion: source_downs[i](s_stft) → source_resblocks[i] → add
        char sd_wn[32], sd_bn[32];
        std::snprintf(sd_wn, sizeof(sd_wn), "s3.v.sd.%d.weight", stage);
        std::snprintf(sd_bn, sizeof(sd_bn), "s3.v.sd.%d.bias", stage);
        ggml_tensor* sd_w = T(c, sd_wn);
        ggml_tensor* sd_b = T(c, sd_bn);
        if (sd_w && s_stft) {
            const int sd_strides[] = {15, 3, 1};
            const int sd_pads[] = {7, 1, 0};
            ggml_tensor* si = ggml_conv_1d(ctx0, sd_w, s_stft, sd_strides[stage], sd_pads[stage], 1);
            if (sd_b)
                si = ggml_add(ctx0, si, ggml_reshape_2d(ctx0, sd_b, 1, (int)sd_b->ne[0]));
            const int srb_kernels[] = {7, 7, 11};
            const int srb_dilations[][3] = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
            ggml_tensor* srb_in = si;
            for (int d = 0; d < 3; d++) {
                char key2[48];
                int dil = srb_dilations[stage][d];
                int k2 = srb_kernels[stage];
                int pad2 = (k2 * dil - dil) / 2;
                std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.a1.%d.alpha", stage, d);
                ggml_tensor* sa1 = T(c, key2);
                if (sa1) {
                    ggml_tensor* a = ggml_reshape_2d(ctx0, sa1, 1, (int)sa1->ne[0]);
                    ggml_tensor* ax = ggml_mul(ctx0, si, a);
                    ggml_tensor* s_ax = ggml_sin(ctx0, ax);
                    si = ggml_add(ctx0, si, ggml_div(ctx0, ggml_mul(ctx0, s_ax, s_ax), a));
                }
                std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.c1.%d.weight", stage, d);
                ggml_tensor* sc1w = T(c, key2);
                std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.c1.%d.bias", stage, d);
                ggml_tensor* sc1b = T(c, key2);
                if (sc1w) {
                    si = ggml_conv_1d(ctx0, sc1w, si, 1, pad2, dil);
                    if (sc1b)
                        si = ggml_add(ctx0, si, ggml_reshape_2d(ctx0, sc1b, 1, (int)sc1b->ne[0]));
                }
                std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.a2.%d.alpha", stage, d);
                ggml_tensor* sa2 = T(c, key2);
                if (sa2) {
                    ggml_tensor* a2 = ggml_reshape_2d(ctx0, sa2, 1, (int)sa2->ne[0]);
                    ggml_tensor* ax2 = ggml_mul(ctx0, si, a2);
                    ggml_tensor* s_ax2 = ggml_sin(ctx0, ax2);
                    si = ggml_add(ctx0, si, ggml_div(ctx0, ggml_mul(ctx0, s_ax2, s_ax2), a2));
                }
                std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.c2.%d.weight", stage, d);
                ggml_tensor* sc2w = T(c, key2);
                std::snprintf(key2, sizeof(key2), "s3.v.srb.%d.c2.%d.bias", stage, d);
                ggml_tensor* sc2b = T(c, key2);
                if (sc2w) {
                    int p2 = (k2 - 1) / 2;
                    si = ggml_conv_1d(ctx0, sc2w, si, 1, p2, 1);
                    if (sc2b)
                        si = ggml_add(ctx0, si, ggml_reshape_2d(ctx0, sc2b, 1, (int)sc2b->ne[0]));
                }
                si = ggml_add(ctx0, si, srb_in);
                srb_in = si;
            }
            int T_x = (int)x->ne[0];
            int T_si = (int)si->ne[0];
            int T_min = std::min(T_x, T_si);
            if (T_si > T_min) {
                si = ggml_view_2d(ctx0, si, T_min, (int)si->ne[1], si->nb[1], 0);
                si = ggml_cont(ctx0, si);
            }
            if (T_x > T_min) {
                x = ggml_view_2d(ctx0, x, T_min, (int)x->ne[1], x->nb[1], 0);
                x = ggml_cont(ctx0, x);
            }
            x = ggml_add(ctx0, x, si);
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
                    ggml_tensor* a = ggml_reshape_2d(ctx0, alpha1, 1, (int)alpha1->ne[0]);
                    ggml_tensor* ax = ggml_mul(ctx0, x, a);
                    ggml_tensor* sin_ax = ggml_sin(ctx0, ax);
                    ggml_tensor* sin2 = ggml_mul(ctx0, sin_ax, sin_ax);
                    ggml_tensor* sin2_over_a = ggml_div(ctx0, sin2, a);
                    x = ggml_add(ctx0, x, sin2_over_a);
                }
                // Debug markers for sub-operations within snake1
                if (stage_dump && stage == 0 && rb == 0) {
                    char dname[48];
                    std::snprintf(dname, sizeof(dname), "voc_rb0k0_snake1_d%d", d);
                    ggml_set_name(x, dname);
                    ggml_set_output(x);
                    if (d == 0) {
                        // Also dump the alpha*x and sin²/alpha intermediates
                        if (alpha1) {
                            ggml_tensor* a = ggml_reshape_2d(ctx0, alpha1, 1, (int)alpha1->ne[0]);
                            ggml_set_name(a, "dbg_alpha_reshaped");
                            ggml_set_output(a);
                        }
                    }
                }

                // Conv1d with dilation
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.c1.%d.weight", rb_idx, d);
                ggml_tensor* c1w = T(c, key);
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.c1.%d.bias", rb_idx, d);
                ggml_tensor* c1b = T(c, key);
                if (c1w) {
                    x = ggml_conv_1d(ctx0, c1w, x, 1, pad, dil);
                    if (c1b)
                        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c1b, 1, (int)c1b->ne[0]));
                }
                if (stage_dump && stage == 0 && rb == 0) {
                    char dname[48];
                    std::snprintf(dname, sizeof(dname), "voc_rb0k0_conv1_d%d", d);
                    ggml_set_name(x, dname);
                    ggml_set_output(x);
                }

                // Snake activation 2
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
                if (stage_dump && stage == 0 && rb == 0) {
                    char dname[48];
                    std::snprintf(dname, sizeof(dname), "voc_rb0k0_snake2_d%d", d);
                    ggml_set_name(x, dname);
                    ggml_set_output(x);
                }

                // Conv2 (dilation=1)
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.c2.%d.weight", rb_idx, d);
                ggml_tensor* c2w = T(c, key);
                std::snprintf(key, sizeof(key), "s3.v.rb.%d.c2.%d.bias", rb_idx, d);
                ggml_tensor* c2b = T(c, key);
                if (c2w) {
                    int pad2 = (k - 1) / 2; // dilation=1, symmetric padding
                    x = ggml_conv_1d(ctx0, c2w, x, 1, pad2, 1);
                    if (c2b)
                        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c2b, 1, (int)c2b->ne[0]));
                }
                if (stage_dump && stage == 0 && rb == 0) {
                    char dname[48];
                    std::snprintf(dname, sizeof(dname), "voc_rb0k0_conv2_d%d", d);
                    ggml_set_name(x, dname);
                    ggml_set_output(x);
                }

                // Residual
                x = ggml_add(ctx0, x, rb_residual);
                rb_residual = x;
                if (stage_dump && stage == 0 && rb == 0) {
                    char dname[48];
                    std::snprintf(dname, sizeof(dname), "voc_rb0k0_res_d%d", d);
                    ggml_set_name(x, dname);
                    ggml_set_output(x);
                }
            }
            // Accumulate for averaging
            if (!rb_sum)
                rb_sum = x;
            else
                rb_sum = ggml_add(ctx0, rb_sum, x);
        }
        // Average the 3 ResBlock outputs
        x = ggml_scale(ctx0, rb_sum, 1.0f / 3.0f);
        {
            char rname[32];
            std::snprintf(rname, sizeof(rname), "voc_rb_%d", stage);
            ggml_set_name(x, rname);
            ggml_set_output(x);
        }
    }

    // LeakyReLU
    x = ggml_leaky_relu(ctx0, x, 0.1f, false);

    // conv_post: Conv1d(64→18, k=7, padding=3)
    ggml_tensor* cpost_w = T(c, "s3.v.cpost.weight");
    ggml_tensor* cpost_b = T(c, "s3.v.cpost.bias");
    if (cpost_w) {
        x = ggml_conv_1d(ctx0, cpost_w, x, 1, 3, 1);
        if (cpost_b)
            x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, cpost_b, 1, (int)cpost_b->ne[0]));
    }

    ggml_set_name(x, "voc_conv_post");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);

    // Execute graph
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "s3gen: failed to alloc vocoder graph\n");
        // Fallback to noise
        return std::vector<float>(T_mel * 480, 0.0f);
    }

    // Set mel input: ggml tensor ne[0]=T, ne[1]=80 → data stored as
    // data[c * T + t] (channel-first, T is the fast axis).
    // Our mel is already channel-first (80, T) → no conversion needed!
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "voc_mel"), mel.data(), 0, mel.size() * sizeof(float));

    // Generate source STFT: F0 prediction → SineGen(9 harmonics) → Linear+tanh → STFT
    // Python: f0 = f0_predictor(mel) → upsample(120x) → SineGen → SourceModuleHnNSF → STFT
    {
        ggml_tensor* src_t = ggml_graph_get_tensor(gf, "source_stft");
        if (src_t) {
            std::vector<float> src_stft((size_t)T_src * 18, 0.0f);
            if (source_stft_cf && T_src_ext > 0) {
                std::memcpy(src_stft.data(), source_stft_cf, src_stft.size() * sizeof(float));
            } else {
                const float sine_amp = 0.1f;
                const float noise_std = 0.003f;
                const int n_harm_plus1 = 9; // fundamental + 8 overtones
                const float sr = 24000.0f;
                const int stft_nfft = istft_nfft;     // 16
                const int stft_hop = istft_hop;       // 4
                const int n_freq = stft_nfft / 2 + 1; // 9
                const int upsample_factor = 120;      // 8*5*3

                // 1. Predict F0 from mel
                std::vector<float> f0 = run_f0_predictor(c, mel, T_mel);
                if (c->verbosity >= 2) {
                    float f0_mean = 0, f0_max = 0;
                    int f0_voiced = 0;
                    for (int t = 0; t < T_mel; t++) {
                        f0_mean += f0[t];
                        if (f0[t] > f0_max)
                            f0_max = f0[t];
                        if (f0[t] > 0)
                            f0_voiced++;
                    }
                    f0_mean /= T_mel;
                    fprintf(stderr, "s3gen: F0 mean=%.1f max=%.1f voiced=%d/%d\n", f0_mean, f0_max, f0_voiced, T_mel);
                }

                // 2. Upsample F0 to audio rate (nearest-neighbor)
                int T_audio = T_mel * upsample_factor * stft_hop;
                std::vector<float> f0_up(T_audio);
                for (int t = 0; t < T_audio; t++)
                    f0_up[t] = f0[t / upsample_factor];

                // 3. SineGen: 9 harmonics with phase accumulation
                uint64_t rng = 54321;
                auto next_u = [&]() -> float {
                    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                    return ((float)(rng >> 33) + 0.5f) / (float)(1ULL << 31);
                };
                auto next_gauss = [&]() -> float {
                    float u1 = next_u(), u2 = next_u();
                    if (u1 < 1e-7f)
                        u1 = 1e-7f;
                    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * (float)M_PI * u2);
                };

                // Random initial phase per harmonic (fundamental=0)
                std::vector<float> phase_offset(n_harm_plus1, 0.0f);
                for (int h = 1; h < n_harm_plus1; h++)
                    phase_offset[h] = (next_u() * 2.0f - 1.0f) * (float)M_PI;

                // Generate sine waves: (T_audio, 9) → voiced: sine+noise, unvoiced: noise
                std::vector<float> sine_waves((size_t)T_audio * n_harm_plus1, 0.0f);
                std::vector<float> cumphase(n_harm_plus1, 0.0f);
                for (int t = 0; t < T_audio; t++) {
                    float f0_val = f0_up[t];
                    bool voiced = (f0_val > 0.0f);
                    float noise_amp = voiced ? noise_std : (sine_amp / 3.0f);
                    for (int h = 0; h < n_harm_plus1; h++) {
                        float freq_norm = f0_val * (float)(h + 1) / sr;
                        cumphase[h] += freq_norm;
                        cumphase[h] -= std::floor(cumphase[h]);
                        float theta = 2.0f * (float)M_PI * cumphase[h] + phase_offset[h];
                        float sine_val = sine_amp * std::sin(theta);
                        float noise_val = noise_amp * next_gauss();
                        sine_waves[(size_t)t * n_harm_plus1 + h] = voiced ? (sine_val + noise_val) : noise_val;
                    }
                }

                // 4. SourceModuleHnNSF: Linear(9→1) + tanh
                ggml_tensor* ms_w = T(c, "s3.v.ms.ll.weight");
                ggml_tensor* ms_b = T(c, "s3.v.ms.ll.bias");
                std::vector<float> ll_w(n_harm_plus1, 0.0f);
                float ll_b = 0.0f;
                if (ms_w)
                    tensor_get_f32(ms_w, ll_w.data(), 0, n_harm_plus1);
                if (ms_b)
                    tensor_get_f32(ms_b, &ll_b, 0, 1);

                std::vector<float> source(T_audio, 0.0f);
                for (int t = 0; t < T_audio; t++) {
                    float val = ll_b;
                    for (int h = 0; h < n_harm_plus1; h++)
                        val += ll_w[h] * sine_waves[(size_t)t * n_harm_plus1 + h];
                    source[t] = std::tanh(val);
                }

                // 5. STFT of source signal (center=True, pad_mode='reflect', Hann window)
                std::vector<float> stft_win(stft_nfft);
                for (int i = 0; i < stft_nfft; i++)
                    stft_win[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / stft_nfft));

                int pad = stft_nfft / 2;
                for (int frame = 0; frame < T_src; frame++) {
                    int center = frame * stft_hop;
                    for (int f = 0; f < n_freq; f++) {
                        float re = 0.0f, im = 0.0f;
                        for (int n = 0; n < stft_nfft; n++) {
                            int src_idx = center - pad + n;
                            if (src_idx < 0)
                                src_idx = -src_idx;
                            else if (src_idx >= T_audio)
                                src_idx = 2 * T_audio - src_idx - 2;
                            src_idx = std::max(0, std::min(src_idx, T_audio - 1));
                            float s = source[src_idx];
                            float w = stft_win[n] * s;
                            float angle = -2.0f * (float)M_PI * f * n / stft_nfft;
                            re += w * std::cos(angle);
                            im += w * std::sin(angle);
                        }
                        src_stft[f * T_src + frame] = re;
                        src_stft[(n_freq + f) * T_src + frame] = im;
                    }
                }
            }
            ggml_backend_tensor_set(src_t, src_stft.data(), 0, src_stft.size() * sizeof(float));
        }
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "s3gen: vocoder compute failed\n");
        return std::vector<float>(T_mel * 480, 0.0f);
    }

    // Dump stage outputs if requested
    if (stage_dump) {
        const char* dump_names[] = {
            "voc_conv_pre",
            "voc_ups_0",
            "voc_rb_0",
            "voc_ups_1",
            "voc_rb_1",
            "voc_ups_2",
            "voc_rb_2",
            "voc_conv_post",
            "voc_rb0k0_snake1_d0",
            "voc_rb0k0_conv1_d0",
            "voc_rb0k0_snake2_d0",
            "voc_rb0k0_conv2_d0",
            "voc_rb0k0_res_d0",
            "voc_rb0k0_snake1_d1",
            "voc_rb0k0_conv1_d1",
            "voc_rb0k0_snake2_d1",
            "voc_rb0k0_conv2_d1",
            "voc_rb0k0_res_d1",
            "voc_rb0k0_snake1_d2",
            "voc_rb0k0_conv1_d2",
            "voc_rb0k0_snake2_d2",
            "voc_rb0k0_conv2_d2",
            "voc_rb0k0_res_d2",
        };
        for (auto& dn : dump_names) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, dn);
            if (t) {
                size_t n = ggml_nelements(t);
                auto& v = (*stage_dump)[dn];
                v.resize(n);
                ggml_backend_tensor_get(t, v.data(), 0, n * sizeof(float));
            }
        }
    }

    // Detailed sub-stage diagnostics for vocoder debugging
    if (stage_dump) {
        auto dump_tensor = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t)
                return;
            int Td = (int)t->ne[0], Cd = (int)t->ne[1];
            std::vector<float> d(Td * Cd);
            ggml_backend_tensor_get(t, d.data(), 0, ggml_nbytes(t));
            float rms_v = 0;
            for (auto v : d)
                rms_v += v * v;
            rms_v = std::sqrt(rms_v / d.size());
            // ne[0]=T, ne[1]=C → data[c*T+t]
            fprintf(stderr, "  %-24s ne=(%d,%d) rms=%.6f (t=0,c=0..4)=[%.4f %.4f %.4f %.4f %.4f]\n", name, Td, Cd,
                    rms_v, d[0 * Td + 0], d[1 * Td + 0], d[2 * Td + 0], d[3 * Td + 0], d[4 * Td + 0]);
        };
        fprintf(stderr, "\n=== Vocoder sub-stage diagnostics ===\n");
        dump_tensor("voc_conv_pre");
        dump_tensor("voc_ups_0");
        for (int d = 0; d < 3; d++) {
            char n[48];
            std::snprintf(n, sizeof(n), "voc_rb0k0_snake1_d%d", d);
            dump_tensor(n);
            std::snprintf(n, sizeof(n), "voc_rb0k0_conv1_d%d", d);
            dump_tensor(n);
            std::snprintf(n, sizeof(n), "voc_rb0k0_snake2_d%d", d);
            dump_tensor(n);
            std::snprintf(n, sizeof(n), "voc_rb0k0_conv2_d%d", d);
            dump_tensor(n);
            std::snprintf(n, sizeof(n), "voc_rb0k0_res_d%d", d);
            dump_tensor(n);
        }
        dump_tensor("voc_rb_0");
        fprintf(stderr, "=== End sub-stage diagnostics ===\n\n");
    }

    // Read conv_pre diagnostic
    ggml_tensor* cpre_out = ggml_graph_get_tensor(gf, "voc_conv_pre");
    if (cpre_out && c->verbosity >= 1) {
        size_t nb = ggml_nbytes(cpre_out);
        std::vector<float> cpre_data(nb / sizeof(float));
        ggml_backend_tensor_get(cpre_out, cpre_data.data(), 0, nb);
        float rms = 0;
        for (auto v : cpre_data)
            rms += v * v;
        rms = std::sqrt(rms / cpre_data.size());
        int T_cpre = (int)cpre_out->ne[0];
        // Read (t=0, c=0..4): ggml stores as ne[0]=T, ne[1]=C → data[c * T + t]
        fprintf(stderr,
                "s3gen: conv_pre ne=(%d,%d) rms=%.3f (t=0,c=0..4)=[%.3f %.3f %.3f %.3f %.3f] (ref: rms=5.50, [0.204 "
                "1.674 1.600 3.275 1.361])\n",
                T_cpre, (int)cpre_out->ne[1], rms, cpre_data[0 * T_cpre + 0], cpre_data[1 * T_cpre + 0],
                cpre_data[2 * T_cpre + 0], cpre_data[3 * T_cpre + 0], cpre_data[4 * T_cpre + 0]);
    }

    // Read STFT output
    ggml_tensor* stft_out = ggml_graph_get_tensor(gf, "voc_conv_post");
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
            if (stft_data[i] > stft_max)
                stft_max = stft_data[i];
            if (stft_data[i] < stft_min)
                stft_min = stft_data[i];
        }
        stft_rms = std::sqrt(stft_rms / stft_data.size());
        fprintf(stderr, "s3gen: STFT values range=[%.3f, %.3f] rms=%.4f (ref: [-1.1, 1.7])\n", stft_min, stft_max,
                stft_rms);
        // Show first frame's 18 channels: data[ch * T_stft + 0]
        fprintf(stderr, "s3gen: STFT frame[0]: ");
        for (int ch = 0; ch < C_stft && ch < 18; ch++)
            fprintf(stderr, "%.3f ", stft_data[ch * T_stft + 0]);
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
        // stft_data layout: ggml ne[0]=T_stft (fast), ne[1]=18 (slow)
        // → data[channel * T_stft + frame]
        // First 9 channels = log-magnitude, last 9 = raw phase input
        float mag[9], ph[9];
        for (int f = 0; f < n_freq; f++) {
            float raw_mag = stft_data[f * T_stft + frame];
            if (raw_mag > 100.0f)
                raw_mag = 100.0f; // clip
            mag[f] = std::exp(raw_mag);
            ph[f] = std::sin(stft_data[(n_freq + f) * T_stft + frame]);
        }

        // Build complex STFT bins: X[f] = mag[f] * (cos(ph[f]) + j*sin(ph[f]))
        float re[9], im[9];
        for (int f = 0; f < n_freq; f++) {
            re[f] = mag[f] * std::cos(ph[f]);
            im[f] = mag[f] * std::sin(ph[f]);
        }

        int start = frame * istft_hop;
        if (c->istft_full_idft) {
            // Full N-point iDFT matching torch.istft: build the complete
            // spectrum via Hermitian symmetry, then standard inverse DFT.
            // X_full[0..N-1] where X_full[k] = conj(X_full[N-k]) for real signals.
            float re_full[16], im_full[16];
            for (int f = 0; f < n_freq; f++) {
                re_full[f] = re[f];
                im_full[f] = im[f];
            }
            // Mirror: X[N-f] = conj(X[f]) for f = 1..N/2-1
            for (int f = 1; f < n_freq - 1; f++) {
                re_full[istft_nfft - f] = re[f];
                im_full[istft_nfft - f] = -im[f];
            }
            // iDFT: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^{j*2π*k*n/N}
            for (int n = 0; n < istft_nfft && (start + n) < n_samples; n++) {
                float sample = 0.0f;
                for (int k = 0; k < istft_nfft; k++) {
                    float angle = 2.0f * (float)M_PI * k * n / (float)istft_nfft;
                    sample += re_full[k] * std::cos(angle) - im_full[k] * std::sin(angle);
                }
                sample /= (float)istft_nfft;
                wav[start + n] += sample * win[n];
                win_sum[start + n] += win[n] * win[n];
            }
        } else {
            // Default: Hermitian-optimized iDFT (fewer ops, same result for real signals)
            for (int n = 0; n < istft_nfft && (start + n) < n_samples; n++) {
                float sample = re[0]; // DC: real part only (imag=0 for real signal at DC)
                for (int f = 1; f < n_freq - 1; f++) {
                    float angle = 2.0f * (float)M_PI * f * n / (float)istft_nfft;
                    sample += 2.0f * (re[f] * std::cos(angle) - im[f] * std::sin(angle));
                }
                // Nyquist (f=N/2): real only, no conjugate pair
                float angle_ny = 2.0f * (float)M_PI * (n_freq - 1) * n / (float)istft_nfft;
                sample += re[n_freq - 1] * std::cos(angle_ny) - im[n_freq - 1] * std::sin(angle_ny);
                sample /= (float)istft_nfft;
                wav[start + n] += sample * win[n];
                win_sum[start + n] += win[n] * win[n];
            }
        }
    }

    // Normalize by COLA (constant overlap-add) condition
    for (int i = 0; i < n_samples; i++) {
        if (win_sum[i] > 1e-8f)
            wav[i] /= win_sum[i];
    }

    // torch.istft(center=True) removes the n_fft/2 padding implied by
    // centered frames, returning exactly hop * (n_frames - 1) samples.
    const int center_pad = istft_nfft / 2;
    int final_len = T_mel * 480;
    std::vector<float> wav_trimmed;
    wav_trimmed.reserve((size_t)final_len);
    for (int i = 0; i < final_len; ++i) {
        int src = center_pad + i;
        float v = (src >= 0 && src < (int)wav.size()) ? wav[src] : 0.0f;
        wav_trimmed.push_back(std::max(-0.99f, std::min(0.99f, v)));
    }

    return wav_trimmed;
}

// ── Full pipeline ───────────────────────────────────────────────

static bool chatterbox_s3gen_compute_gen_mel(struct chatterbox_s3gen_context* ctx, const int32_t* speech_tokens,
                                             int n_speech_tokens, const int32_t* prompt_tokens, int n_prompt_tokens,
                                             const float* prompt_feat, int prompt_feat_len,
                                             const float* spk_embedding, int n_cfm_steps,
                                             std::vector<float>& gen_mel_out, int* out_T_mel) {
    if (!ctx || !speech_tokens || n_speech_tokens <= 0)
        return false;
    if (out_T_mel)
        *out_T_mel = 0;
    if (n_cfm_steps <= 0)
        n_cfm_steps = 10;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: %d speech tokens + %d prompt tokens, %d CFM steps\n", n_speech_tokens, n_prompt_tokens,
                n_cfm_steps);
    }

    // Check for stage dump mode (per-stage intermediate comparison)
    const char* dump_env = std::getenv("CRISPASR_S3GEN_DUMP");
    bool dump_stages = dump_env && dump_env[0] == '1';

    // 1. Conformer encoder: tokens → (80, T_mel)
    std::vector<float> h = run_conformer_encoder(ctx, speech_tokens, n_speech_tokens, prompt_tokens, n_prompt_tokens);

    int T_mel_total = (n_prompt_tokens + n_speech_tokens) * 2; // 2x upsample
    int T_mel_prompt = n_prompt_tokens * 2;
    int T_mel_gen = n_speech_tokens * 2;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: encoder output T_mel=%d (prompt=%d, gen=%d)\n", T_mel_total, T_mel_prompt, T_mel_gen);
    }

    if (dump_stages) {
        // Dump encoder output for diff comparison
        // h is (80, T_mel_total) channel-first: data[ch * T + t]
        float enc_rms = 0;
        for (auto v : h)
            enc_rms += v * v;
        enc_rms = std::sqrt(enc_rms / h.size());
        fprintf(stderr, "s3gen[dump]: encoder_out (%d, %d) rms=%.4f\n", 80, T_mel_total, enc_rms);
        // Print first few values: (t=0, ch=0..4)
        fprintf(stderr, "s3gen[dump]: encoder_out (t=0,ch=0..4): ");
        for (int ch = 0; ch < 5; ch++)
            fprintf(stderr, "%.4f ", h[ch * T_mel_total + 0]);
        fprintf(stderr, "\n");
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
        tensor_get_f32(spk_w, sw.data(), 0, sw.size());
        if (spk_b)
            tensor_get_f32(spk_b, sb.data(), 0, sb.size());

        // Normalize embedding (L2 norm)
        float norm = 0.0f;
        for (int i = 0; i < 192; i++)
            norm += spk_embedding[i] * spk_embedding[i];
        norm = std::sqrt(norm + 1e-12f);

        for (int i = 0; i < 80; i++) {
            float sum = sb[i];
            for (int j = 0; j < 192; j++) {
                sum += sw[i * 192 + j] * (spk_embedding[j] / norm);
            }
            spk_proj[i] = sum;
        }
    }

    if (dump_stages) {
        fprintf(stderr, "s3gen[dump]: spk_proj first 5: [%.6f %.6f %.6f %.6f %.6f]\n", spk_proj[0], spk_proj[1],
                spk_proj[2], spk_proj[3], spk_proj[4]);
        // Dump conditioning mel rms
        float cond_rms = 0;
        for (auto v : cond)
            cond_rms += v * v;
        cond_rms = std::sqrt(cond_rms / cond.size());
        fprintf(stderr, "s3gen[dump]: cond rms=%.4f\n", cond_rms);
    }

    // Diagnostic: compare encoder and spk_proj with Python reference
    if (ctx->verbosity >= 2) {
        float enc_rms = 0;
        for (auto v : h)
            enc_rms += v * v;
        enc_rms = std::sqrt(enc_rms / h.size());
        fprintf(stderr, "s3gen: encoder_out rms=%.4f (ref: 0.4602)\n", enc_rms);
        fprintf(stderr,
                "s3gen: spk_proj first 5: [%.4f %.4f %.4f %.4f %.4f] (ref: [0.0964 0.0206 0.0434 -0.0960 "
                "0.2453])\n",
                spk_proj[0], spk_proj[1], spk_proj[2], spk_proj[3], spk_proj[4]);
    }

    // 4. CFM Euler solver: noise → mel
    // Detect meanflow: time_embed_mixer weight exists only in meanflow models
    bool is_meanflow = (T(ctx, "s3.fd.tmx.weight") != nullptr);
    float cfg = is_meanflow ? 0.0f : 0.7f;
    // Meanflow defaults to 2 CFM steps (distilled model); override if caller passed default 10
    int actual_steps = n_cfm_steps;
    if (is_meanflow && n_cfm_steps == 10) {
        actual_steps = 2; // Python default for meanflow
    }
    if (ctx->verbosity >= 1 && is_meanflow) {
        fprintf(stderr, "s3gen: meanflow mode (%d steps, linear schedule, no CFG)\n", actual_steps);
    }
    std::vector<float> mel =
        cfm_euler_solve(ctx, h, cond, spk_proj, T_mel_total, actual_steps, cfg, is_meanflow, dump_stages);

    // 5. Extract generated portion (skip prompt region)
    std::vector<float> gen_mel(80 * T_mel_gen);
    for (int b = 0; b < 80; b++) {
        std::memcpy(&gen_mel[b * T_mel_gen], &mel[b * T_mel_total + T_mel_prompt], T_mel_gen * sizeof(float));
    }

    if (dump_stages) {
        float gen_rms = 0;
        for (auto v : gen_mel)
            gen_rms += v * v;
        gen_rms = std::sqrt(gen_rms / gen_mel.size());
        fprintf(stderr, "s3gen[dump]: gen_mel (%d, %d) rms=%.4f\n", 80, T_mel_gen, gen_rms);
        fprintf(stderr, "s3gen[dump]: gen_mel (t=0,ch=0..4): ");
        for (int ch = 0; ch < 5; ch++)
            fprintf(stderr, "%.4f ", gen_mel[ch * T_mel_gen + 0]);
        fprintf(stderr, "\n");
    }

    gen_mel_out = std::move(gen_mel);
    if (out_T_mel)
        *out_T_mel = T_mel_gen;
    return true;
}

extern "C" float* chatterbox_s3gen_synthesize_mel(struct chatterbox_s3gen_context* ctx, const int32_t* speech_tokens,
                                                  int n_speech_tokens, const int32_t* prompt_tokens,
                                                  int n_prompt_tokens, const float* prompt_feat, int prompt_feat_len,
                                                  const float* spk_embedding, int n_cfm_steps, int* out_T_mel) {
    if (!ctx || !speech_tokens || n_speech_tokens <= 0 || !out_T_mel)
        return nullptr;
    *out_T_mel = 0;
    std::vector<float> gen_mel;
    if (!chatterbox_s3gen_compute_gen_mel(ctx, speech_tokens, n_speech_tokens, prompt_tokens, n_prompt_tokens,
                                          prompt_feat, prompt_feat_len, spk_embedding, n_cfm_steps, gen_mel,
                                          out_T_mel)) {
        return nullptr;
    }
    float* out = (float*)malloc(gen_mel.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, gen_mel.data(), gen_mel.size() * sizeof(float));
    return out;
}

static void apply_trim_fade(std::vector<float>& wav) {
    const int n_trim = 24000 / 50; // 20 ms = half of a frame
    const int fade_len = 2 * n_trim;
    const int n = std::min<int>((int)wav.size(), fade_len);
    for (int i = 0; i < n; ++i) {
        float mul = 0.0f;
        if (i >= n_trim) {
            const float x = (float)(i - n_trim) / (float)std::max(1, n_trim - 1);
            mul = (std::cos((1.0f - x) * (float)M_PI) + 1.0f) * 0.5f;
        }
        wav[(size_t)i] *= mul;
    }
}

extern "C" float* chatterbox_s3gen_synthesize(struct chatterbox_s3gen_context* ctx, const int32_t* speech_tokens,
                                              int n_speech_tokens, const int32_t* prompt_tokens, int n_prompt_tokens,
                                              const float* prompt_feat, int prompt_feat_len, const float* spk_embedding,
                                              int n_cfm_steps, int* out_n_samples) {
    if (!ctx || !speech_tokens || n_speech_tokens <= 0 || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    std::vector<float> gen_mel;
    int T_mel_gen = 0;
    if (!chatterbox_s3gen_compute_gen_mel(ctx, speech_tokens, n_speech_tokens, prompt_tokens, n_prompt_tokens,
                                          prompt_feat, prompt_feat_len, spk_embedding, n_cfm_steps, gen_mel,
                                          &T_mel_gen)) {
        return nullptr;
    }

    // 6. Vocoder: mel → waveform
    std::vector<float> wav = hift_vocoder_cpu(ctx, gen_mel, T_mel_gen);
    apply_trim_fade(wav);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: generated %zu samples (%.2f sec @ 24kHz)\n", wav.size(), (float)wav.size() / 24000.0f);
    }

    // Copy to malloc'd buffer
    float* out = (float*)malloc(wav.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, wav.data(), wav.size() * sizeof(float));
    *out_n_samples = (int)wav.size();
    return out;
}

extern "C" float* chatterbox_s3gen_vocode(struct chatterbox_s3gen_context* ctx, const float* mel_cf, int T_mel,
                                          int* out_n_samples) {
    return chatterbox_s3gen_vocode_with_source_stft(ctx, mel_cf, T_mel, nullptr, 0, out_n_samples);
}

extern "C" float* chatterbox_s3gen_vocode_with_source_stft(struct chatterbox_s3gen_context* ctx, const float* mel_cf,
                                                           int T_mel, const float* source_stft_cf, int T_src,
                                                           int* out_n_samples) {
    if (!ctx || !mel_cf || T_mel <= 0 || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    std::vector<float> mel(mel_cf, mel_cf + 80 * T_mel);
    std::vector<float> wav = hift_vocoder_cpu(ctx, mel, T_mel, source_stft_cf, T_src);
    apply_trim_fade(wav);

    if (wav.empty())
        return nullptr;
    float* out = (float*)malloc(wav.size() * sizeof(float));
    std::memcpy(out, wav.data(), wav.size() * sizeof(float));
    *out_n_samples = (int)wav.size();
    return out;
}

extern "C" float* chatterbox_s3gen_vocode_dump(struct chatterbox_s3gen_context* ctx, const float* mel_cf, int T_mel,
                                               int* out_n_samples, const char** stage_names, float** stage_data,
                                               int* stage_sizes, int n_stages) {
    return chatterbox_s3gen_vocode_dump_with_source_stft(ctx, mel_cf, T_mel, nullptr, 0, out_n_samples, stage_names,
                                                         stage_data, stage_sizes, n_stages);
}

extern "C" float* chatterbox_s3gen_vocode_dump_with_source_stft(struct chatterbox_s3gen_context* ctx, const float* mel_cf,
                                                                int T_mel, const float* source_stft_cf, int T_src,
                                                                int* out_n_samples, const char** stage_names,
                                                                float** stage_data, int* stage_sizes, int n_stages) {
    if (!ctx || !mel_cf || T_mel <= 0 || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    std::map<std::string, std::vector<float>> dump;
    std::vector<float> mel(mel_cf, mel_cf + 80 * T_mel);
    std::vector<float> wav = hift_vocoder_cpu(ctx, mel, T_mel, source_stft_cf, T_src, &dump);
    apply_trim_fade(wav);

    // Fill caller's stage arrays
    for (int i = 0; i < n_stages; i++) {
        auto it = dump.find(stage_names[i]);
        if (it != dump.end()) {
            stage_sizes[i] = (int)it->second.size();
            stage_data[i] = (float*)malloc(it->second.size() * sizeof(float));
            std::memcpy(stage_data[i], it->second.data(), it->second.size() * sizeof(float));
        } else {
            stage_sizes[i] = 0;
            stage_data[i] = nullptr;
        }
    }

    if (wav.empty())
        return nullptr;
    float* out = (float*)malloc(wav.size() * sizeof(float));
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
