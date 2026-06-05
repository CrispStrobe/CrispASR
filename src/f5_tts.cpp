// f5_tts.cpp — native ggml runtime for SWivid/F5-TTS.
//
// Architecture (all F32 on CPU, F16 for large weight storage):
//   1. TextEmbedding: char embedding(2546, 512) + sinusoidal pos +
//      4 ConvNeXtV2 blocks (512-d, intermediate 1024)
//   2. InputEmbedding: Linear(712, 1024) concat(x, cond, text) +
//      ConvPositionEmbedding (2× Conv1d k=31 groups=16 + Mish)
//   3. TimestepEmbedding: sinusoidal(256) → Linear(256,1024) → SiLU → Linear(1024,1024)
//   4. DiT: 22 blocks, each:
//      - AdaLN-Zero: SiLU(t) → Linear(1024, 6144) → split 6× (shift/scale/gate for attn+mlp)
//      - Self-attention: Q/K/V proj → RoPE → scaled_dot_product → O proj
//        (16 heads, dim_head=64, bidirectional, no mask for batch=1)
//      - Gated residual: x = x + gate_msa * attn_out
//      - Modulated LayerNorm: ff_norm(x) * (1 + scale_mlp) + shift_mlp
//      - FFN: Linear(1024, 2048) → GELU_tanh → Linear(2048, 1024)
//      - Gated residual: x = x + gate_mlp * ff_out
//   5. AdaLN-Final + Linear(1024, 100) → velocity prediction
//   6. Euler ODE solver: 32 steps with EPSS timesteps + sway + CFG
//   7. Vocos vocoder: Conv1d(100,512,k7) → LN → 8× ConvNeXt(512,1536) →
//      LN → Linear(512, 1026) → split mag+phase → iSTFT → 24 kHz PCM
//
// The implementation uses per-module ggml sub-graphs. Each module builds
// its own graph, computes, and returns CPU-side results.

#include "f5_tts.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Hyperparameters ──────────────────────────────────────────────

struct f5_hparams {
    int dim = 1024;
    int depth = 22;
    int heads = 16;
    int dim_head = 64;
    int ff_mult = 2;
    int text_dim = 512;
    int text_num_embeds = 2546;
    int conv_layers = 4;
    int mel_dim = 100;
    int sample_rate = 24000;
    int hop_length = 256;
    int win_length = 1024;
    int n_fft = 1024;
    int ode_steps = 32;
    float cfg_strength = 2.0f;
    float sway_coef = -1.0f;
    int conv_pos_kernel = 31;
    int conv_pos_groups = 16;
    // Vocos
    int voc_dim = 512;
    int voc_intermediate_dim = 1536;
    int voc_num_layers = 8;
    int voc_n_fft = 1024;
    int voc_hop_length = 256;
};

// ── Weight structure ─────────────────────────────────────────────

struct f5_weights {
    // Text embedding
    ggml_tensor* text_emb_weight = nullptr; // (2546, 512)

    // Text ConvNeXtV2 blocks (4×)
    struct text_block {
        ggml_tensor* dw_weight; // (512, 1, 7) depthwise conv
        ggml_tensor* dw_bias;
        ggml_tensor* norm_weight;
        ggml_tensor* norm_bias;
        ggml_tensor* pw_up_weight; // (1024, 512)
        ggml_tensor* pw_up_bias;
        ggml_tensor* pw_down_weight; // (512, 1024)
        ggml_tensor* pw_down_bias;
        ggml_tensor* grn_gamma; // (1, 1, 1024)
        ggml_tensor* grn_beta;
    };
    std::vector<text_block> text_blocks;

    // Timestep embedding
    ggml_tensor* time_mlp_0_weight; // (1024, 256)
    ggml_tensor* time_mlp_0_bias;
    ggml_tensor* time_mlp_1_weight; // (1024, 1024)
    ggml_tensor* time_mlp_1_bias;

    // Input embedding
    ggml_tensor* input_proj_weight; // (1024, 712)
    ggml_tensor* input_proj_bias;
    ggml_tensor* conv_pos_0_weight; // (1024, 64, 31)
    ggml_tensor* conv_pos_0_bias;
    ggml_tensor* conv_pos_1_weight;
    ggml_tensor* conv_pos_1_bias;

    // DiT blocks (22×)
    struct dit_block {
        // AdaLN
        ggml_tensor* adaln_weight; // (6144, 1024)
        ggml_tensor* adaln_bias;
        // Self-attention
        ggml_tensor* attn_q_weight;
        ggml_tensor* attn_q_bias;
        ggml_tensor* attn_k_weight;
        ggml_tensor* attn_k_bias;
        ggml_tensor* attn_v_weight;
        ggml_tensor* attn_v_bias;
        ggml_tensor* attn_o_weight;
        ggml_tensor* attn_o_bias;
        // FFN
        ggml_tensor* ffn_up_weight; // (2048, 1024)
        ggml_tensor* ffn_up_bias;
        ggml_tensor* ffn_down_weight; // (1024, 2048)
        ggml_tensor* ffn_down_bias;
    };
    std::vector<dit_block> dit_blocks;

    // Final norm + proj
    ggml_tensor* final_adaln_weight; // (2048, 1024)
    ggml_tensor* final_adaln_bias;
    ggml_tensor* final_proj_weight; // (100, 1024)
    ggml_tensor* final_proj_bias;

    // Rotary
    ggml_tensor* rotary_inv_freq; // (32,)

    // Vocos
    ggml_tensor* voc_embed_weight; // (512, 100, 7)
    ggml_tensor* voc_embed_bias;
    ggml_tensor* voc_norm_weight;
    ggml_tensor* voc_norm_bias;
    struct vocos_block {
        ggml_tensor* dw_weight;
        ggml_tensor* dw_bias;
        ggml_tensor* norm_weight;
        ggml_tensor* norm_bias;
        ggml_tensor* pw_up_weight;
        ggml_tensor* pw_up_bias;
        ggml_tensor* pw_down_weight;
        ggml_tensor* pw_down_bias;
        ggml_tensor* layer_scale; // (512,)
    };
    std::vector<vocos_block> voc_blocks;
    ggml_tensor* voc_final_norm_weight;
    ggml_tensor* voc_final_norm_bias;
    ggml_tensor* voc_head_weight; // (1026, 512)
    ggml_tensor* voc_head_bias;
};

// ── Vocab ────────────────────────────────────────────────────────

struct f5_vocab {
    std::vector<std::string> chars;
    std::map<std::string, int> char_to_idx;
};

// ── Context ──────────────────────────────────────────────────────

struct f5_tts_context {
    f5_hparams hp;
    f5_weights w;
    f5_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;

    // Runtime params
    int seed;
    int ode_steps;
    float cfg_strength;
    float sway_coef;
    float speed;
    int verbosity;
    int n_threads;
    std::string dump_dir;

    // Reference audio state
    std::vector<float> ref_mel; // (T_ref, mel_dim) row-major
    int ref_mel_T = 0;
    std::string ref_text;

    // Diff harness: inject reference initial noise for reproducibility
    std::vector<float> ref_init_noise;
};

// ── Diff dump helpers ────────────────────────────────────────────

static void dump_stage(const f5_tts_context* ctx, const char* label, const float* data, size_t n) {
    if (ctx->dump_dir.empty())
        return;
    std::string path = ctx->dump_dir + "/" + label + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, sizeof(float), n, f);
        fclose(f);
    }
}

// ── Mini graph helper ────────────────────────────────────────────

struct f5_mini_graph {
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;

    f5_mini_graph(ggml_backend_sched_t s, size_t ctx_size = 32 * 1024 * 1024) : sched(s) {
        struct ggml_init_params params = {ctx_size, nullptr, true};
        ctx = ggml_init(params);
    }
    ~f5_mini_graph() {
        if (ctx)
            ggml_free(ctx);
    }

    std::vector<float> compute(ggml_tensor* output, int /*n_threads*/) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, output);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "f5_tts: graph alloc failed\n");
            return {};
        }
        ggml_backend_sched_graph_compute(sched, gf);
        int n = (int)ggml_nelements(output);
        std::vector<float> result(n);
        ggml_backend_tensor_get(output, result.data(), 0, n * sizeof(float));
        return result;
    }

    bool compute_into(ggml_tensor* output, float* dst, size_t nbytes, int /*n_threads*/) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, output);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "f5_tts: graph alloc failed\n");
            return false;
        }
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_tensor_get(output, dst, 0, nbytes);
        return true;
    }

    // Compute graph returning multiple outputs
    bool compute_multi(std::vector<ggml_tensor*>& outputs, int /*n_threads*/) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        for (auto* o : outputs) {
            ggml_build_forward_expand(gf, o);
        }
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "f5_tts: graph alloc failed\n");
            return false;
        }
        ggml_backend_sched_graph_compute(sched, gf);
        return true;
    }

    void set_input(ggml_tensor* t, const void* data, size_t nbytes) { ggml_backend_tensor_set(t, data, 0, nbytes); }
};

// ── Tensor read helper ───────────────────────────────────────────

static void read_tensor_f32(ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    const size_t nbytes = ggml_nbytes(t);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
    } else {
        // F16 or quantized (Q4_K, Q8_0, etc.) — dequantize via type traits.
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const auto to_float = ggml_get_type_traits(t->type)->to_float;
        if (to_float) {
            to_float(raw.data(), out.data(), n);
        } else {
            fprintf(stderr, "f5_tts: unsupported tensor type %d for '%s'\n", (int)t->type, t->name);
            std::fill(out.begin(), out.end(), 0.0f);
        }
    }
}

// ── sinusoidal position embedding ────────────────────────────────

static void sinusoidal_pos_emb(int dim, int max_len, float* out) {
    // out: (max_len, dim) row-major
    // Matches precompute_freqs_cis: freqs_cos ++ freqs_sin
    int half_dim = dim / 2;
    float theta = 10000.0f;
    for (int t = 0; t < max_len; t++) {
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = (float)t * freq;
            out[t * dim + d] = cosf(angle);            // cos part
            out[t * dim + d + half_dim] = sinf(angle); // sin part
        }
    }
}

// ── Timestep embedding (sinusoidal + MLP) ────────────────────────

static std::vector<float> compute_time_embed(f5_tts_context* ctx, float t_val) {
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;

    // Sinusoidal position embedding for timestep
    int freq_dim = 256;
    int half = freq_dim / 2;
    std::vector<float> sinus(freq_dim);
    float scale = 1000.0f;
    for (int d = 0; d < half; d++) {
        float emb_val = logf(10000.0f) / (float)(half - 1);
        float freq = expf(-(float)d * emb_val);
        float angle = scale * t_val * freq;
        sinus[d] = sinf(angle);
        sinus[d + half] = cosf(angle);
    }

    // MLP: Linear(256→1024) → SiLU → Linear(1024→1024)
    f5_mini_graph mg(ctx->sched);
    ggml_tensor* inp = ggml_new_tensor_1d(mg.ctx, GGML_TYPE_F32, freq_dim);
    ggml_set_name(inp, "time_sinus");
    ggml_set_input(inp);

    // Layer 0: Linear + SiLU
    ggml_tensor* h = ggml_mul_mat(mg.ctx, w.time_mlp_0_weight, inp);
    h = ggml_add(mg.ctx, h, w.time_mlp_0_bias);
    h = ggml_silu(mg.ctx, h);

    // Layer 1: Linear
    h = ggml_mul_mat(mg.ctx, w.time_mlp_1_weight, h);
    h = ggml_add(mg.ctx, h, w.time_mlp_1_bias);
    ggml_set_name(h, "time_emb");
    ggml_set_output(h);

    ggml_cgraph* gf = ggml_new_graph_custom(mg.ctx, 256, false);
    ggml_build_forward_expand(gf, h);
    ggml_backend_sched_reset(mg.sched);
    if (!ggml_backend_sched_alloc_graph(mg.sched, gf))
        return {};
    mg.set_input(inp, sinus.data(), freq_dim * sizeof(float));
    ggml_backend_sched_graph_compute(mg.sched, gf);

    std::vector<float> result(hp.dim);
    ggml_backend_tensor_get(h, result.data(), 0, hp.dim * sizeof(float));
    return result;
}

// ── Text tokenization ────────────────────────────────────────────

static std::vector<int32_t> tokenize_text(const f5_vocab& vocab, const std::string& text) {
    // Character-level tokenization matching list_str_to_idx
    std::vector<int32_t> tokens;
    size_t i = 0;
    while (i < text.size()) {
        // Try to match multi-byte UTF-8 characters
        int len = 1;
        if ((text[i] & 0x80) == 0)
            len = 1;
        else if ((text[i] & 0xE0) == 0xC0)
            len = 2;
        else if ((text[i] & 0xF0) == 0xE0)
            len = 3;
        else if ((text[i] & 0xF8) == 0xF0)
            len = 4;
        if (i + len > text.size())
            break;

        std::string ch = text.substr(i, len);
        auto it = vocab.char_to_idx.find(ch);
        if (it != vocab.char_to_idx.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(0); // unknown → 0
        }
        i += len;
    }
    return tokens;
}

// ── pinyin conversion (simplified, ASCII-only passthrough) ───────
// For a full implementation, would need rjieba + pypinyin equivalent.
// For now, pass ASCII text through character-by-character.

static std::vector<std::string> convert_to_pinyin(const std::string& text) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.size()) {
        int len = 1;
        if ((text[i] & 0x80) == 0)
            len = 1;
        else if ((text[i] & 0xE0) == 0xC0)
            len = 2;
        else if ((text[i] & 0xF0) == 0xE0)
            len = 3;
        else if ((text[i] & 0xF8) == 0xF0)
            len = 4;
        if (i + len > text.size())
            break;
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

// ── Text Encoder (embedding + sinusoidal pos + ConvNeXtV2 blocks) ─

static std::vector<float> compute_text_embed(f5_tts_context* ctx, const int32_t* tokens, int n_tokens, int seq_len) {
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;

    // tokens are in range [-1, vocab_size-1]. We add 1 to make 0 the filler.
    // Then pad/truncate to seq_len.
    std::vector<int32_t> padded(seq_len, 0);
    for (int i = 0; i < std::min(n_tokens, seq_len); i++) {
        padded[i] = tokens[i] + 1; // shift by 1 (filler = 0)
    }

    // Create text mask: 1 where padded == 0 (filler/padding)
    std::vector<float> text_mask(seq_len);
    for (int i = 0; i < seq_len; i++) {
        text_mask[i] = (padded[i] == 0) ? 1.0f : 0.0f;
    }

    // Embedding lookup (done manually since ggml embedding is int-indexed)
    std::vector<float> emb_weight;
    read_tensor_f32(w.text_emb_weight, emb_weight);

    std::vector<float> text_emb(seq_len * hp.text_dim, 0.0f);
    for (int t = 0; t < seq_len; t++) {
        int idx = padded[t];
        if (idx >= 0 && idx < hp.text_num_embeds) {
            for (int d = 0; d < hp.text_dim; d++) {
                text_emb[t * hp.text_dim + d] = emb_weight[idx * hp.text_dim + d];
            }
        }
    }

    // Add sinusoidal position embedding (precompute_freqs_cis)
    // Only for positions within valid range
    std::vector<float> freqs(seq_len * hp.text_dim);
    sinusoidal_pos_emb(hp.text_dim, seq_len, freqs.data());
    // Mask positions beyond valid tokens
    for (int t = 0; t < seq_len; t++) {
        if (t < n_tokens) { // valid position
            for (int d = 0; d < hp.text_dim; d++) {
                text_emb[t * hp.text_dim + d] += freqs[t * hp.text_dim + d];
            }
        }
    }

    // Apply text mask (zero out filler positions)
    for (int t = 0; t < seq_len; t++) {
        if (text_mask[t] > 0.5f) {
            for (int d = 0; d < hp.text_dim; d++) {
                text_emb[t * hp.text_dim + d] = 0.0f;
            }
        }
    }

    // ConvNeXtV2 blocks — implemented on CPU for exact semantics.
    // Each block: dwconv(k=7) → LN → pw_up → GELU → GRN → pw_down → residual
    for (int b = 0; b < hp.conv_layers; b++) {
        const auto& blk = w.text_blocks[b];
        int D = hp.text_dim;
        int T = seq_len;
        int K = 7, pad = 3;
        int inter_dim = D * 2; // intermediate_dim = text_dim * conv_mult = 512 * 2

        // Read weights
        std::vector<float> dw_w, dw_b, norm_w, norm_b, pw_up_w, pw_up_b;
        std::vector<float> pw_down_w, pw_down_b, grn_g, grn_b_v;
        read_tensor_f32(blk.dw_weight, dw_w); // (D, 1, K) → flat (D*K)
        read_tensor_f32(blk.dw_bias, dw_b);
        read_tensor_f32(blk.norm_weight, norm_w);
        read_tensor_f32(blk.norm_bias, norm_b);
        read_tensor_f32(blk.pw_up_weight, pw_up_w); // (inter_dim, D)
        read_tensor_f32(blk.pw_up_bias, pw_up_b);
        read_tensor_f32(blk.pw_down_weight, pw_down_w); // (D, inter_dim)
        read_tensor_f32(blk.pw_down_bias, pw_down_b);
        read_tensor_f32(blk.grn_gamma, grn_g); // (1,1,inter_dim)
        read_tensor_f32(blk.grn_beta, grn_b_v);

        // text_emb is (T, D) row-major
        std::vector<float> residual_v = text_emb;

        // 1. Depthwise conv (groups=D, kernel=7, pad=3, dilation=1)
        // For each channel c, convolve text_emb[:,c] with dw_w[c*K..(c+1)*K-1]
        std::vector<float> conv_out(T * D, 0.0f);
        for (int c = 0; c < D; c++) {
            for (int t = 0; t < T; t++) {
                float sum = dw_b[c];
                for (int k = 0; k < K; k++) {
                    int ti = t + k - pad;
                    if (ti >= 0 && ti < T) {
                        sum += text_emb[ti * D + c] * dw_w[c * K + k];
                    }
                }
                conv_out[t * D + c] = sum;
            }
        }

        // 2. LayerNorm (over D dimension, per time step)
        for (int t = 0; t < T; t++) {
            float mean = 0, var = 0;
            for (int d = 0; d < D; d++)
                mean += conv_out[t * D + d];
            mean /= D;
            for (int d = 0; d < D; d++) {
                float diff = conv_out[t * D + d] - mean;
                var += diff * diff;
            }
            var /= D;
            float inv_std = 1.0f / sqrtf(var + 1e-6f);
            for (int d = 0; d < D; d++) {
                conv_out[t * D + d] = ((conv_out[t * D + d] - mean) * inv_std) * norm_w[d] + norm_b[d];
            }
        }

        // 3. Pointwise up: (T, D) × (inter_dim, D)^T → (T, inter_dim)
        std::vector<float> up_out(T * inter_dim, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int o = 0; o < inter_dim; o++) {
                float sum = pw_up_b[o];
                for (int d = 0; d < D; d++) {
                    sum += conv_out[t * D + d] * pw_up_w[o * D + d];
                }
                up_out[t * inter_dim + o] = sum;
            }
        }

        // 4. GELU (exact: x * 0.5 * (1 + erf(x/sqrt(2))))
        for (auto& v : up_out) {
            v = v * 0.5f * (1.0f + erff(v / sqrtf(2.0f)));
        }

        // 5. GRN: Gx = L2_norm(x, dim=T), Nx = Gx / (mean(Gx, dim=D) + eps)
        // out = gamma * (x * Nx) + beta + x
        // Gx[t, d] = ||x[:, d]||_2 (norm across T for each feature d)
        // Wait, the PyTorch code does: Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        // In (B, T, D) layout, dim=1 is T. So Gx = L2 norm across T → shape (B, 1, D)
        std::vector<float> gx(inter_dim, 0.0f);
        for (int d = 0; d < inter_dim; d++) {
            float sum_sq = 0;
            for (int t = 0; t < T; t++) {
                float v = up_out[t * inter_dim + d];
                sum_sq += v * v;
            }
            gx[d] = sqrtf(sum_sq);
        }
        // Nx = Gx / (mean(Gx, dim=-1) + eps)  → mean over D
        float gx_mean = 0;
        for (int d = 0; d < inter_dim; d++)
            gx_mean += gx[d];
        gx_mean /= inter_dim;
        std::vector<float> nx(inter_dim);
        for (int d = 0; d < inter_dim; d++)
            nx[d] = gx[d] / (gx_mean + 1e-6f);
        // out = gamma * (x * Nx) + beta + x
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < inter_dim; d++) {
                float x_val = up_out[t * inter_dim + d];
                up_out[t * inter_dim + d] = grn_g[d] * (x_val * nx[d]) + grn_b_v[d] + x_val;
            }
        }

        // 6. Pointwise down: (T, inter_dim) × (D, inter_dim)^T → (T, D)
        std::vector<float> down_out(T * D, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int o = 0; o < D; o++) {
                float sum = pw_down_b[o];
                for (int d = 0; d < inter_dim; d++) {
                    sum += up_out[t * inter_dim + d] * pw_down_w[o * inter_dim + d];
                }
                down_out[t * D + o] = sum;
            }
        }

        // 7. Residual
        for (int i = 0; i < T * D; i++) {
            text_emb[i] = residual_v[i] + down_out[i];
        }

        // Apply text mask
        for (int t = 0; t < T; t++) {
            if (text_mask[t] > 0.5f) {
                for (int d = 0; d < D; d++)
                    text_emb[t * D + d] = 0.0f;
            }
        }
    }

    return text_emb;
}

// ── Radix-2 FFT (reused by mel + ISTFT) ─────────────────────────

// In-place radix-2 FFT. sign=+1 for forward, -1 for inverse.
// Arrays re[] and im[] must have length N (power of 2).
// For inverse, caller must divide by N afterwards.
static void fft_radix2(float* re, float* im, int N, int sign) {
    int log2n = 0;
    for (int tmp = N; tmp > 1; tmp >>= 1)
        log2n++;
    // Bit-reversal permutation
    for (int i = 0; i < N; i++) {
        int j = 0, x = i;
        for (int b = 0; b < log2n; b++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    // Butterfly passes
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int half = m / 2;
        float wpr = cosf(2.0f * (float)M_PI / (float)m);
        float wpi = (float)sign * sinf(2.0f * (float)M_PI / (float)m);
        for (int k = 0; k < N; k += m) {
            float wr = 1.0f, wi = 0.0f;
            for (int j = 0; j < half; j++) {
                float tr = wr * re[k + j + half] - wi * im[k + j + half];
                float ti = wr * im[k + j + half] + wi * re[k + j + half];
                re[k + j + half] = re[k + j] - tr;
                im[k + j + half] = im[k + j] - ti;
                re[k + j] += tr;
                im[k + j] += ti;
                float wt = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = wt;
            }
        }
    }
}

// ── HTK mel scale ───────────────────────────────────────────────

static float hz_to_mel(float f) {
    return 2595.0f * log10f(1.0f + f / 700.0f);
}
static float mel_to_hz(float m) {
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

// ── Mel spectrogram (vocos-type, 24kHz) ──────────────────────────
//
// Matches torchaudio.transforms.MelSpectrogram with:
//   sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256,
//   n_mels=100, power=1, center=True, normalized=False, norm=None
// Followed by: clamp(min=1e-5).log()
//
// Returns (T, n_mels) row-major mel spectrogram.

static std::vector<float> compute_mel_spectrogram(const float* pcm_24k, int n_samples, int n_fft, int hop_length,
                                                  int win_length, int n_mels, int& T_out) {
    int n_freqs = n_fft / 2 + 1; // 513
    float sr = 24000.0f;

    // ── Build mel filterbank (n_freqs × n_mels) ──
    std::vector<float> mel_fb(n_freqs * n_mels, 0.0f);
    {
        float f_min = 0.0f, f_max = sr / 2.0f;
        float mel_min = hz_to_mel(f_min);
        float mel_max = hz_to_mel(f_max);
        std::vector<float> mel_points(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++)
            mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * (float)i / (float)(n_mels + 1));
        for (int m = 0; m < n_mels; m++) {
            float f_left = mel_points[m];
            float f_center = mel_points[m + 1];
            float f_right = mel_points[m + 2];
            for (int k = 0; k < n_freqs; k++) {
                float f = (float)k * sr / (float)n_fft;
                float val = 0.0f;
                if (f >= f_left && f <= f_center && f_center > f_left)
                    val = (f - f_left) / (f_center - f_left);
                else if (f > f_center && f <= f_right && f_right > f_center)
                    val = (f_right - f) / (f_right - f_center);
                mel_fb[k * n_mels + m] = val;
            }
        }
    }

    // ── Hann window ──
    std::vector<float> hann(win_length);
    for (int i = 0; i < win_length; i++)
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)win_length));

    // ── Center padding (reflect) ──
    int pad = n_fft / 2;
    int padded_len = n_samples + 2 * pad;
    std::vector<float> padded(padded_len);
    // Reflect pad left
    for (int i = 0; i < pad; i++)
        padded[i] = pcm_24k[pad - i]; // reflect: index 1,2,3,...,pad
    // Copy signal
    for (int i = 0; i < n_samples; i++)
        padded[pad + i] = pcm_24k[i];
    // Reflect pad right
    for (int i = 0; i < pad; i++)
        padded[pad + n_samples + i] = pcm_24k[n_samples - 2 - i]; // reflect

    // ── STFT frames ──
    int T = (padded_len - n_fft) / hop_length + 1;
    T_out = T;

    // ── Process each frame: window → FFT → magnitude → mel → log ──
    std::vector<float> mel_spec(T * n_mels);
    std::vector<float> frame_re(n_fft), frame_im(n_fft);
    std::vector<float> mag(n_freqs);

    for (int t = 0; t < T; t++) {
        int offset = t * hop_length;

        // Window the frame
        for (int i = 0; i < n_fft; i++) {
            frame_re[i] = padded[offset + i] * hann[i];
            frame_im[i] = 0.0f;
        }

        // Forward FFT (sign = -1 for forward DFT convention e^{-j2pi})
        fft_radix2(frame_re.data(), frame_im.data(), n_fft, -1);

        // Magnitude spectrum (power=1)
        for (int k = 0; k < n_freqs; k++) {
            mag[k] = sqrtf(frame_re[k] * frame_re[k] + frame_im[k] * frame_im[k]);
        }

        // Mel filterbank multiplication: mel[m] = sum_k mag[k] * fb[k, m]
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_freqs; k++)
                sum += mag[k] * mel_fb[k * n_mels + m];
            // Clamp and log
            if (sum < 1e-5f)
                sum = 1e-5f;
            mel_spec[t * n_mels + m] = logf(sum);
        }
    }

    return mel_spec;
}

// ── DiT forward (one ODE step) ───────────────────────────────────
//
// Runs the full DiT: input_embed → 22 blocks → final adaln + proj.
// Returns velocity prediction (T, mel_dim).
//
// x:       (T, mel_dim)   — current ODE state
// cond:    (T, mel_dim)   — conditioning (masked ref mel)
// text:    (T, text_dim)  — text embedding
// time_emb: (dim,)        — timestep embedding

static std::vector<float> dit_forward(f5_tts_context* ctx, const float* x_data, int T, int mel_dim,
                                      const float* cond_data, const float* text_data, int text_dim,
                                      const float* time_emb_data, bool drop_audio_cond, bool drop_text, int step_idx) {
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    int dim = hp.dim;

    // ── InputEmbedding: cat(x, cond, text) → proj → +conv_pos_embed ──
    // Concatenate along feature dim: (T, mel_dim + mel_dim + text_dim) = (T, 712)
    int cat_dim = mel_dim + mel_dim + text_dim;
    std::vector<float> cat_input(T * cat_dim);
    for (int t = 0; t < T; t++) {
        // x
        for (int d = 0; d < mel_dim; d++)
            cat_input[t * cat_dim + d] = x_data[t * mel_dim + d];
        // cond (zero if drop_audio_cond)
        for (int d = 0; d < mel_dim; d++)
            cat_input[t * cat_dim + mel_dim + d] = drop_audio_cond ? 0.0f : cond_data[t * mel_dim + d];
        // text (zero if drop_text — but text is already zeroed during embedding if drop_text)
        for (int d = 0; d < text_dim; d++)
            cat_input[t * cat_dim + mel_dim + mel_dim + d] = text_data[t * text_dim + d];
    }

    if (step_idx == 0 && !drop_audio_cond) {
        dump_stage(ctx, "cat_input", cat_input.data(), cat_input.size());
        // Dump the weight matrix for verification
        std::vector<float> proj_w;
        read_tensor_f32(w.input_proj_weight, proj_w);
        dump_stage(ctx, "debug_proj_weight", proj_w.data(), proj_w.size());
    }

    // Linear projection: (T, 712) → (T, 1024) — on CPU for exact results
    std::vector<float> proj_w, proj_b;
    read_tensor_f32(w.input_proj_weight, proj_w); // (1024, 712) row-major
    read_tensor_f32(w.input_proj_bias, proj_b);

    std::vector<float> hidden(T * dim, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < dim; o++) {
            float sum = proj_b[o];
            for (int i = 0; i < cat_dim; i++) {
                sum += cat_input[t * cat_dim + i] * proj_w[o * cat_dim + i];
            }
            hidden[t * dim + o] = sum;
        }
    }

    if (step_idx == 0 && !drop_audio_cond) {
        dump_stage(ctx, "input_proj_out", hidden.data(), hidden.size());
    }

    // ConvPositionEmbedding: 2× (Conv1d(dim, dim, k=31, g=16, p=15) + Mish)
    // Implemented on CPU as grouped conv (groups=16).
    {
        int K = 31, pad_k = 15, groups = 16;
        int ch_per_group = dim / groups; // 1024/16 = 64

        auto grouped_conv_mish = [&](const std::vector<float>& input, ggml_tensor* w_tensor, ggml_tensor* b_tensor) {
            std::vector<float> wt, bias;
            read_tensor_f32(w_tensor, wt);
            read_tensor_f32(b_tensor, bias);

            std::vector<float> output(T * dim, 0.0f);
            for (int g = 0; g < groups; g++) {
                int ch_start = g * ch_per_group;
                for (int oc = ch_start; oc < ch_start + ch_per_group; oc++) {
                    for (int t = 0; t < T; t++) {
                        float sum = bias[oc];
                        for (int ic_local = 0; ic_local < ch_per_group; ic_local++) {
                            int ic = ch_start + ic_local;
                            for (int k = 0; k < K; k++) {
                                int ti = t + k - pad_k;
                                if (ti >= 0 && ti < T) {
                                    float w_val = wt[oc * ch_per_group * K + ic_local * K + k];
                                    sum += input[ti * dim + ic] * w_val;
                                }
                            }
                        }
                        output[t * dim + oc] = sum;
                    }
                }
            }
            for (auto& v : output) {
                float sp = logf(1.0f + expf(v));
                v = v * tanhf(sp);
            }
            return output;
        };

        std::vector<float> proj_out = hidden;
        hidden = grouped_conv_mish(hidden, w.conv_pos_0_weight, w.conv_pos_0_bias);
        hidden = grouped_conv_mish(hidden, w.conv_pos_1_weight, w.conv_pos_1_bias);

        for (size_t i = 0; i < hidden.size(); i++) {
            hidden[i] += proj_out[i];
        }
    }

    if (step_idx == 0 && !drop_audio_cond) {
        dump_stage(ctx, "input_embed", hidden.data(), hidden.size());
    }

    // ── RoPE note ──
    // x_transformers rotate_half pairs adjacent elements (2k, 2k+1):
    //   x = rearrange(x, '(d r) -> d r', r=2)  →  x1=x[0::2], x2=x[1::2]
    //   rotated = stack(-x2, x1) → [-x1, x0, -x3, x2, ...]
    // RotaryEmbedding interleaves freqs: stack((f,f), dim=-1) → [f0,f0,f1,f1,...]
    // so paired elements share the SAME frequency — a standard 2D rotation,
    // equivalent to GGML_ROPE_TYPE_NORMAL (mode=0).

    // ── 22 DiT blocks ──
    for (int blk_idx = 0; blk_idx < hp.depth; blk_idx++) {
        const auto& blk = w.dit_blocks[blk_idx];

        // Each block:
        // 1. adaln_modulation = linear(silu(time_emb)) → split into 6 × dim
        // 2. norm(x) * (1 + scale_msa) + shift_msa → attn input
        // 3. self_attn with RoPE → attn_output
        // 4. x = x + gate_msa * attn_output
        // 5. ff_norm(x) * (1 + scale_mlp) + shift_mlp → ffn input
        // 6. ffn(input) → ffn_output
        // 7. x = x + gate_mlp * ffn_output

        int n_heads = hp.heads;
        int head_dim = hp.dim_head;

        // ── Unified ggml graph: AdaLN + QKV + RoPE + Attention + O-proj + FFN ──
        {
            f5_mini_graph mg(ctx->sched, 128 * 1024 * 1024);

            ggml_tensor* x_in = ggml_new_tensor_2d(mg.ctx, GGML_TYPE_F32, dim, T);
            ggml_set_name(x_in, "blk_in");
            ggml_set_input(x_in);

            ggml_tensor* t_emb = ggml_new_tensor_1d(mg.ctx, GGML_TYPE_F32, dim);
            ggml_set_name(t_emb, "t_emb");
            ggml_set_input(t_emb);

            // Position indices for RoPE: [0, 1, 2, ..., T-1]
            ggml_tensor* pos = ggml_new_tensor_1d(mg.ctx, GGML_TYPE_I32, T);
            ggml_set_name(pos, "pos");
            ggml_set_input(pos);

            // ── AdaLN modulation ──
            ggml_tensor* emb = ggml_silu(mg.ctx, t_emb);
            emb = ggml_mul_mat(mg.ctx, blk.adaln_weight, emb);
            emb = ggml_add(mg.ctx, emb, blk.adaln_bias);

            ggml_tensor* shift_msa = ggml_view_1d(mg.ctx, emb, dim, 0 * dim * sizeof(float));
            ggml_tensor* scale_msa = ggml_view_1d(mg.ctx, emb, dim, 1 * dim * sizeof(float));
            ggml_tensor* gate_msa = ggml_view_1d(mg.ctx, emb, dim, 2 * dim * sizeof(float));
            ggml_tensor* shift_mlp = ggml_view_1d(mg.ctx, emb, dim, 3 * dim * sizeof(float));
            ggml_tensor* scale_mlp = ggml_view_1d(mg.ctx, emb, dim, 4 * dim * sizeof(float));
            ggml_tensor* gate_mlp = ggml_view_1d(mg.ctx, emb, dim, 5 * dim * sizeof(float));

            // ── Pre-norm (AdaLN) ──
            ggml_tensor* norm_x = ggml_norm(mg.ctx, x_in, 1e-6f);
            ggml_tensor* scaled = ggml_mul(mg.ctx, norm_x, scale_msa);
            norm_x = ggml_add(mg.ctx, norm_x, scaled);
            norm_x = ggml_add(mg.ctx, norm_x, shift_msa);

            // ── QKV projections ──
            ggml_tensor* q = ggml_mul_mat(mg.ctx, blk.attn_q_weight, norm_x);
            q = ggml_add(mg.ctx, q, blk.attn_q_bias);
            ggml_tensor* k = ggml_mul_mat(mg.ctx, blk.attn_k_weight, norm_x);
            k = ggml_add(mg.ctx, k, blk.attn_k_bias);
            ggml_tensor* v = ggml_mul_mat(mg.ctx, blk.attn_v_weight, norm_x);
            v = ggml_add(mg.ctx, v, blk.attn_v_bias);

            // ── Reshape to (head_dim, n_heads, T) for RoPE + attention ──
            // ggml_rope expects ne[2]=T (sequence dim) to match positions.
            // ggml_flash_attn_ext expects (head_dim, T, n_heads) so we
            // permute after RoPE.
            q = ggml_reshape_3d(mg.ctx, q, head_dim, n_heads, T);
            k = ggml_reshape_3d(mg.ctx, k, head_dim, n_heads, T);
            v = ggml_reshape_3d(mg.ctx, v, head_dim, n_heads, T);

            // ── RoPE (GGML_ROPE_TYPE_NORMAL = mode 0, freq_base=10000) ──
            // Input shape: (head_dim, n_heads, T). RoPE rotates along ne[0],
            // positions indexed by ne[2].
            q = ggml_rope_ext(mg.ctx, q, pos, nullptr, head_dim, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(mg.ctx, k, pos, nullptr, head_dim, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // ── Permute for flash_attn: (head_dim, n_heads, T) → (head_dim, T, n_heads) ──
            q = ggml_permute(mg.ctx, q, 0, 2, 1, 3); // ne: (head_dim, T, n_heads)
            k = ggml_permute(mg.ctx, k, 0, 2, 1, 3);
            v = ggml_permute(mg.ctx, v, 0, 2, 1, 3);

            // ── Flash attention (bidirectional, no mask) ──
            float attn_scale = 1.0f / sqrtf((float)head_dim);
            ggml_tensor* attn_out = ggml_flash_attn_ext(mg.ctx, q, k, v, nullptr, attn_scale, 0.0f, 0.0f);

            // ── Reshape back to (dim, T) ──
            attn_out = ggml_reshape_2d(mg.ctx, attn_out, dim, T);

            // ── O-proj + gated residual ──
            ggml_tensor* attn_proj = ggml_mul_mat(mg.ctx, blk.attn_o_weight, attn_out);
            attn_proj = ggml_add(mg.ctx, attn_proj, blk.attn_o_bias);
            ggml_tensor* gated_attn = ggml_mul(mg.ctx, attn_proj, gate_msa);
            ggml_tensor* x_res = ggml_add(mg.ctx, x_in, gated_attn);

            // ── FFN pre-norm ──
            ggml_tensor* ff_norm = ggml_norm(mg.ctx, x_res, 1e-6f);
            ggml_tensor* ff_scaled = ggml_mul(mg.ctx, ff_norm, scale_mlp);
            ff_norm = ggml_add(mg.ctx, ff_norm, ff_scaled);
            ff_norm = ggml_add(mg.ctx, ff_norm, shift_mlp);

            // ── FFN ──
            ggml_tensor* ff = ggml_mul_mat(mg.ctx, blk.ffn_up_weight, ff_norm);
            ff = ggml_add(mg.ctx, ff, blk.ffn_up_bias);
            ff = ggml_gelu(mg.ctx, ff);
            ff = ggml_mul_mat(mg.ctx, blk.ffn_down_weight, ff);
            ff = ggml_add(mg.ctx, ff, blk.ffn_down_bias);

            // ── Gated residual ──
            ggml_tensor* gated_ff = ggml_mul(mg.ctx, ff, gate_mlp);
            ggml_tensor* x_out = ggml_add(mg.ctx, x_res, gated_ff);
            ggml_set_name(x_out, "blk_out");
            ggml_set_output(x_out);

            // Build and compute
            ggml_cgraph* gf = ggml_new_graph_custom(mg.ctx, 32768, false);
            ggml_build_forward_expand(gf, x_out);
            ggml_backend_sched_reset(mg.sched);
            if (!ggml_backend_sched_alloc_graph(mg.sched, gf))
                return {};

            // Set inputs
            mg.set_input(x_in, hidden.data(), hidden.size() * sizeof(float));
            mg.set_input(t_emb, time_emb_data, dim * sizeof(float));
            std::vector<int32_t> pos_data(T);
            for (int i = 0; i < T; i++)
                pos_data[i] = i;
            mg.set_input(pos, pos_data.data(), T * sizeof(int32_t));

            ggml_backend_sched_graph_compute(mg.sched, gf);

            hidden.resize(T * dim);
            ggml_backend_tensor_get(x_out, hidden.data(), 0, hidden.size() * sizeof(float));
        }
        if (step_idx == 0 && !drop_audio_cond) {
            char label[64];
            snprintf(label, sizeof(label), "dit_layer_%d", blk_idx);
            dump_stage(ctx, label, hidden.data(), hidden.size());
        }
    }

    // ── Final AdaLN + projection ──
    {
        f5_mini_graph mg(ctx->sched);
        ggml_tensor* x_in = ggml_new_tensor_2d(mg.ctx, GGML_TYPE_F32, dim, T);
        ggml_set_name(x_in, "final_in");
        ggml_set_input(x_in);

        ggml_tensor* t_emb = ggml_new_tensor_1d(mg.ctx, GGML_TYPE_F32, dim);
        ggml_set_name(t_emb, "t_emb_final");
        ggml_set_input(t_emb);

        // AdaLN_Final: silu(emb) → linear(dim→2*dim) → split scale, shift
        ggml_tensor* emb = ggml_silu(mg.ctx, t_emb);
        emb = ggml_mul_mat(mg.ctx, w.final_adaln_weight, emb);
        emb = ggml_add(mg.ctx, emb, w.final_adaln_bias); // (2*dim,)

        ggml_tensor* scale = ggml_view_1d(mg.ctx, emb, dim, 0);
        ggml_tensor* shift = ggml_view_1d(mg.ctx, emb, dim, dim * sizeof(float));

        // norm(x) * (1 + scale) + shift = norm + norm*scale + shift
        ggml_tensor* norm_x = ggml_norm(mg.ctx, x_in, 1e-6f);
        ggml_tensor* fn_scaled = ggml_mul(mg.ctx, norm_x, scale);
        norm_x = ggml_add(mg.ctx, norm_x, fn_scaled);
        norm_x = ggml_add(mg.ctx, norm_x, shift);

        // Project to mel_dim
        ggml_tensor* output = ggml_mul_mat(mg.ctx, w.final_proj_weight, norm_x);
        output = ggml_add(mg.ctx, output, w.final_proj_bias);

        ggml_set_name(output, "dit_output");
        ggml_set_output(output);

        ggml_cgraph* gf = ggml_new_graph_custom(mg.ctx, 4096, false);
        ggml_build_forward_expand(gf, output);
        ggml_backend_sched_reset(mg.sched);
        if (!ggml_backend_sched_alloc_graph(mg.sched, gf))
            return {};
        mg.set_input(x_in, hidden.data(), hidden.size() * sizeof(float));
        mg.set_input(t_emb, time_emb_data, dim * sizeof(float));
        ggml_backend_sched_graph_compute(mg.sched, gf);

        std::vector<float> velocity(T * mel_dim);
        ggml_backend_tensor_get(output, velocity.data(), 0, velocity.size() * sizeof(float));

        if (step_idx == 0 && !drop_audio_cond) {
            dump_stage(ctx, "dit_output", velocity.data(), velocity.size());
        }
        return velocity;
    }
}

// ── Vocos vocoder ────────────────────────────────────────────────

// ── CPU Conv1d helper (standard or depthwise) ──────────────────────
// input/output: (C, T) row-major. weight: (C_out, C_in_per_group, K).
static void cpu_conv1d(const float* input, int C_in, int T, const float* weight, const float* bias, int C_out, int K,
                       int pad, int groups, float* output) {
    int ch_per_group_in = C_in / groups;
    int ch_per_group_out = C_out / groups;
    for (int g = 0; g < groups; g++) {
        int oc_start = g * ch_per_group_out;
        int ic_start = g * ch_per_group_in;
        for (int oc = oc_start; oc < oc_start + ch_per_group_out; oc++) {
            for (int t = 0; t < T; t++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic_local = 0; ic_local < ch_per_group_in; ic_local++) {
                    int ic = ic_start + ic_local;
                    for (int k = 0; k < K; k++) {
                        int ti = t + k - pad;
                        if (ti >= 0 && ti < T) {
                            int w_idx = oc * ch_per_group_in * K + ic_local * K + k;
                            sum += input[ic * T + ti] * weight[w_idx];
                        }
                    }
                }
                output[oc * T + t] = sum;
            }
        }
    }
}

// ── CPU LayerNorm over last dim of (T, D) data ─────────────────────
static void cpu_layer_norm(float* data, int T, int D, const float* gamma, const float* beta, float eps) {
    for (int t = 0; t < T; t++) {
        float* row = data + t * D;
        float mean = 0;
        for (int d = 0; d < D; d++)
            mean += row[d];
        mean /= D;
        float var = 0;
        for (int d = 0; d < D; d++) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        var /= D;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < D; d++) {
            row[d] = ((row[d] - mean) * inv_std) * gamma[d] + beta[d];
        }
    }
}

// ── Transpose (C, T) ↔ (T, C) ──────────────────────────────────────
static void transpose_ct(const float* src, int C, int T, float* dst) {
    // src: (C, T) → dst: (T, C)
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            dst[t * C + c] = src[c * T + t];
}

static void transpose_tc(const float* src, int T, int C, float* dst) {
    // src: (T, C) → dst: (C, T)
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            dst[c * T + t] = src[t * C + c];
}

static std::vector<float> vocos_decode(f5_tts_context* ctx, const float* mel_data, int T_mel, int mel_dim) {
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    int D = hp.voc_dim; // 512
    int T = T_mel;
    int inter_dim = hp.voc_intermediate_dim; // 1536

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: vocos_decode T=%d mel_dim=%d voc_dim=%d\n", T, mel_dim, D);
    }

    // mel_data is (T, 100) row-major. Convert to (100, T) for conv operations.
    std::vector<float> x_ct(mel_dim * T);
    transpose_tc(mel_data, T, mel_dim, x_ct.data());

    // ── Embed: Conv1d(100, 512, k=7, pad=3) ──
    {
        std::vector<float> emb_w, emb_b;
        read_tensor_f32(w.voc_embed_weight, emb_w); // (512, 100, 7)
        read_tensor_f32(w.voc_embed_bias, emb_b);
        std::vector<float> out(D * T, 0.0f);
        cpu_conv1d(x_ct.data(), mel_dim, T, emb_w.data(), emb_b.data(), D, 7, 3, 1, out.data());
        x_ct = std::move(out);
    }

    // ── Initial LayerNorm: transpose to (T, D), norm, transpose back ──
    {
        std::vector<float> x_td(T * D);
        transpose_ct(x_ct.data(), D, T, x_td.data());
        std::vector<float> norm_w, norm_b;
        read_tensor_f32(w.voc_norm_weight, norm_w);
        read_tensor_f32(w.voc_norm_bias, norm_b);
        cpu_layer_norm(x_td.data(), T, D, norm_w.data(), norm_b.data(), 1e-6f);
        transpose_tc(x_td.data(), T, D, x_ct.data());
    }

    // ── 8× ConvNeXt blocks ──
    for (int b = 0; b < hp.voc_num_layers; b++) {
        const auto& blk = w.voc_blocks[b];
        std::vector<float> residual = x_ct; // (D, T)

        // 1. Depthwise Conv1d(512, 512, k=7, pad=3, groups=512)
        {
            std::vector<float> dw_w, dw_b;
            read_tensor_f32(blk.dw_weight, dw_w); // (512, 1, 7)
            read_tensor_f32(blk.dw_bias, dw_b);
            std::vector<float> out(D * T, 0.0f);
            cpu_conv1d(x_ct.data(), D, T, dw_w.data(), dw_b.data(), D, 7, 3, D, out.data());
            x_ct = std::move(out);
        }

        // 2. Transpose to (T, D), LayerNorm
        std::vector<float> x_td(T * D);
        transpose_ct(x_ct.data(), D, T, x_td.data());
        {
            std::vector<float> norm_w, norm_b;
            read_tensor_f32(blk.norm_weight, norm_w);
            read_tensor_f32(blk.norm_bias, norm_b);
            cpu_layer_norm(x_td.data(), T, D, norm_w.data(), norm_b.data(), 1e-6f);
        }

        // 3. Pointwise up: Linear(512, 1536)
        std::vector<float> up_out(T * inter_dim, 0.0f);
        {
            std::vector<float> pw_w, pw_b;
            read_tensor_f32(blk.pw_up_weight, pw_w); // (1536, 512)
            read_tensor_f32(blk.pw_up_bias, pw_b);
            for (int t = 0; t < T; t++) {
                for (int o = 0; o < inter_dim; o++) {
                    float sum = pw_b[o];
                    for (int d = 0; d < D; d++)
                        sum += x_td[t * D + d] * pw_w[o * D + d];
                    up_out[t * inter_dim + o] = sum;
                }
            }
        }

        // 4. GELU (exact: x * 0.5 * (1 + erf(x/sqrt(2))))
        for (auto& v : up_out) {
            v = v * 0.5f * (1.0f + erff(v / sqrtf(2.0f)));
        }

        // 5. Pointwise down: Linear(1536, 512)
        std::vector<float> down_out(T * D, 0.0f);
        {
            std::vector<float> pw_w, pw_b;
            read_tensor_f32(blk.pw_down_weight, pw_w); // (512, 1536)
            read_tensor_f32(blk.pw_down_bias, pw_b);
            for (int t = 0; t < T; t++) {
                for (int o = 0; o < D; o++) {
                    float sum = pw_b[o];
                    for (int d = 0; d < inter_dim; d++)
                        sum += up_out[t * inter_dim + d] * pw_w[o * inter_dim + d];
                    down_out[t * D + o] = sum;
                }
            }
        }

        // 6. Layer scale (gamma)
        {
            std::vector<float> gamma;
            read_tensor_f32(blk.layer_scale, gamma); // (512,)
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    down_out[t * D + d] *= gamma[d];
        }

        // 7. Transpose back to (D, T) and add residual
        transpose_tc(down_out.data(), T, D, x_ct.data());
        for (size_t i = 0; i < x_ct.size(); i++)
            x_ct[i] += residual[i];
    }

    // ── Final LayerNorm ──
    std::vector<float> x_td(T * D);
    transpose_ct(x_ct.data(), D, T, x_td.data());
    {
        std::vector<float> norm_w, norm_b;
        read_tensor_f32(w.voc_final_norm_weight, norm_w);
        read_tensor_f32(w.voc_final_norm_bias, norm_b);
        cpu_layer_norm(x_td.data(), T, D, norm_w.data(), norm_b.data(), 1e-6f);
    }
    // x_td is now (T, D) = backbone output

    // ── ISTFTHead: Linear(512, 1026) → split mag/phase → iSTFT ──
    int n_fft = hp.voc_n_fft;    // 1024
    int hop = hp.voc_hop_length; // 256
    int n_freqs = n_fft / 2 + 1; // 513
    int out_dim = n_fft + 2;     // 1026

    // Linear projection
    std::vector<float> head_out(T * out_dim);
    {
        std::vector<float> head_w, head_b;
        read_tensor_f32(w.voc_head_weight, head_w); // (1026, 512)
        read_tensor_f32(w.voc_head_bias, head_b);
        for (int t = 0; t < T; t++) {
            for (int o = 0; o < out_dim; o++) {
                float sum = head_b[o];
                for (int d = 0; d < D; d++)
                    sum += x_td[t * D + d] * head_w[o * D + d];
                head_out[t * out_dim + o] = sum;
            }
        }
    }

    // Split into magnitude and phase, each (T, 513)
    // head_out is (T, 1026) row-major. First 513 = magnitude, last 513 = phase.
    // mag = exp(mag), clip to max 100
    // S = mag * (cos(phase) + i*sin(phase))  → complex (T, 513)
    // Then iSTFT.

    // Transpose to (n_freqs, T) for ISTFT: S_real and S_imag
    std::vector<float> S_real(n_freqs * T), S_imag(n_freqs * T);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < n_freqs; f++) {
            float mag = expf(head_out[t * out_dim + f]);
            if (mag > 100.0f)
                mag = 100.0f;
            float phase = head_out[t * out_dim + n_freqs + f];
            S_real[f * T + t] = mag * cosf(phase);
            S_imag[f * T + t] = mag * sinf(phase);
        }
    }

    // ── ISTFT (custom "same" padding) ──
    // n_fft=1024, hop=256, win_length=1024
    int pad = (n_fft - hop) / 2; // 384
    int output_size = (T - 1) * hop + n_fft;

    // Hann window
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; i++) {
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)n_fft));
    }

    // IRFFT + window + overlap-add
    std::vector<float> audio(output_size, 0.0f);
    std::vector<float> window_env(output_size, 0.0f);

    // For each time frame, compute IRFFT and overlap-add
    std::vector<float> frame_real(n_fft), frame_imag(n_fft);
    for (int t = 0; t < T; t++) {
        // Build full complex spectrum for IRFFT: n_fft complex values
        // We have n_freqs = n_fft/2+1 positive frequency bins.
        // Negative frequencies are conjugate of positive (real signal).
        for (int f = 0; f < n_freqs; f++) {
            frame_real[f] = S_real[f * T + t];
            frame_imag[f] = S_imag[f * T + t];
        }
        // Mirror: for f = n_fft/2+1 .. n_fft-1, X[f] = conj(X[n_fft-f])
        for (int f = n_freqs; f < n_fft; f++) {
            int mirror = n_fft - f;
            frame_real[f] = frame_real[mirror];
            frame_imag[f] = -frame_imag[mirror];
        }

        // IFFT via shared radix-2 (sign=+1 for inverse)
        int N = n_fft;
        // fft_radix2 is in-place, so copy into frame arrays
        fft_radix2(frame_real.data(), frame_imag.data(), N, +1);
        // Divide by N and apply window + overlap-add
        int out_start = t * hop;
        float inv_n = 1.0f / (float)N;
        for (int n = 0; n < N; n++) {
            float sample = frame_real[n] * inv_n; // real part only
            sample *= hann[n];
            audio[out_start + n] += sample;
            window_env[out_start + n] += hann[n] * hann[n];
        }
    }

    // Normalize by window envelope and trim padding
    int trimmed_len = output_size - 2 * pad;
    if (trimmed_len <= 0)
        return {};
    std::vector<float> result(trimmed_len);
    for (int i = 0; i < trimmed_len; i++) {
        float env = window_env[pad + i];
        result[i] = (env > 1e-11f) ? audio[pad + i] / env : 0.0f;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: vocos_decode produced %d samples\n", trimmed_len);
    }
    return result;
}

// ── EPSS timestep schedule ───────────────────────────────────────

static std::vector<float> get_epss_timesteps(int n_steps) {
    float dt = 1.0f / 32.0f;
    // Predefined non-uniform timestep schedules
    if (n_steps == 32) {
        // Uniform for 32 steps
        std::vector<float> t(33);
        for (int i = 0; i <= 32; i++)
            t[i] = (float)i / 32.0f;
        return t;
    }
    // Known schedules from F5-TTS
    std::vector<int> steps;
    switch (n_steps) {
    case 5:
        steps = {0, 2, 4, 8, 16, 32};
        break;
    case 6:
        steps = {0, 2, 4, 6, 8, 16, 32};
        break;
    case 7:
        steps = {0, 2, 4, 6, 8, 16, 24, 32};
        break;
    case 10:
        steps = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
        break;
    case 12:
        steps = {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32};
        break;
    case 16:
        steps = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
        break;
    default: {
        // Linear fallback
        std::vector<float> t(n_steps + 1);
        for (int i = 0; i <= n_steps; i++)
            t[i] = (float)i / (float)n_steps;
        return t;
    }
    }
    std::vector<float> t(steps.size());
    for (size_t i = 0; i < steps.size(); i++)
        t[i] = dt * (float)steps[i];
    return t;
}

// ── ODE Euler solver ─────────────────────────────────────────────

static std::vector<float> euler_solve(f5_tts_context* ctx,
                                      const std::vector<float>& cond,            // (T, mel_dim)
                                      const std::vector<float>& text_emb,        // (T, text_dim)
                                      const std::vector<float>& text_emb_uncond, // (T, text_dim) zeroed
                                      int T, int mel_dim, int text_dim) {
    const auto& hp = ctx->hp;

    // Timestep schedule
    std::vector<float> t_schedule = get_epss_timesteps(ctx->ode_steps);

    // Apply sway sampling coefficient
    if (ctx->sway_coef != 0.0f) {
        for (auto& t : t_schedule) {
            t = t + ctx->sway_coef * (cosf((float)M_PI / 2.0f * t) - 1.0f + t);
        }
    }

    int n_steps = (int)t_schedule.size() - 1;

    // Initial noise y0 ~ N(0, 1)
    std::vector<float> x(T * mel_dim);
    if (!ctx->ref_init_noise.empty() && (int)ctx->ref_init_noise.size() == T * mel_dim) {
        // Use injected reference noise for diff validation
        x = ctx->ref_init_noise;
    } else {
        std::mt19937 rng(ctx->seed ? ctx->seed : std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : x)
            v = dist(rng);
    }

    dump_stage(ctx, "ode_step_0", x.data(), x.size());

    // Euler integration
    for (int step = 0; step < n_steps; step++) {
        float t_val = t_schedule[step];
        float dt = t_schedule[step + 1] - t_val;

        // Compute time embedding for this step
        auto time_emb = compute_time_embed(ctx, t_val);
        if (time_emb.empty())
            return {};

        if (step == 0) {
            dump_stage(ctx, "time_embed", time_emb.data(), time_emb.size());
        }

        if (ctx->cfg_strength < 1e-5f) {
            // No CFG: single forward pass
            auto velocity = dit_forward(ctx, x.data(), T, mel_dim, cond.data(), text_emb.data(), text_dim,
                                        time_emb.data(), false, false, step);
            if (velocity.empty())
                return {};

            for (size_t i = 0; i < x.size(); i++) {
                x[i] += velocity[i] * dt;
            }
        } else {
            // CFG: conditioned + unconditioned forward
            auto v_cond = dit_forward(ctx, x.data(), T, mel_dim, cond.data(), text_emb.data(), text_dim,
                                      time_emb.data(), false, false, step);
            auto v_uncond = dit_forward(ctx, x.data(), T, mel_dim, cond.data(), text_emb_uncond.data(), text_dim,
                                        time_emb.data(), true, true, step);
            if (v_cond.empty() || v_uncond.empty())
                return {};

            // CFG: v = v_cond + cfg * (v_cond - v_uncond)
            float cfg = ctx->cfg_strength;
            for (size_t i = 0; i < x.size(); i++) {
                float v = v_cond[i] + cfg * (v_cond[i] - v_uncond[i]);
                x[i] += v * dt;
            }
        }

        // Dump selected ODE steps
        char label[64];
        snprintf(label, sizeof(label), "ode_step_%d", step + 1);
        dump_stage(ctx, label, x.data(), x.size());

        if (ctx->verbosity >= 2) {
            fprintf(stderr, "  ODE step %d/%d  t=%.4f  dt=%.4f\n", step + 1, n_steps, t_val, dt);
        }
    }

    // Apply conditioning mask: replace ref positions with original cond
    // x = where(cond_mask, cond, x)
    // cond_mask: True for ref positions (where cond != 0)
    for (int t = 0; t < T; t++) {
        bool is_ref = false;
        for (int d = 0; d < mel_dim && !is_ref; d++) {
            if (fabsf(cond[t * mel_dim + d]) > 1e-10f)
                is_ref = true;
        }
        if (is_ref) {
            for (int d = 0; d < mel_dim; d++) {
                x[t * mel_dim + d] = cond[t * mel_dim + d];
            }
        }
    }

    return x;
}

// ── Weight loading ───────────────────────────────────────────────

static bool load_weights(f5_tts_context* ctx, const char* path) {
    auto& hp = ctx->hp;
    auto& w = ctx->w;

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;

    hp.dim = core_gguf::kv_i32(meta, "f5.dim", hp.dim);
    hp.depth = core_gguf::kv_i32(meta, "f5.depth", hp.depth);
    hp.heads = core_gguf::kv_i32(meta, "f5.heads", hp.heads);
    hp.dim_head = core_gguf::kv_i32(meta, "f5.dim_head", hp.dim_head);
    hp.ff_mult = core_gguf::kv_i32(meta, "f5.ff_mult", hp.ff_mult);
    hp.text_dim = core_gguf::kv_i32(meta, "f5.text_dim", hp.text_dim);
    hp.text_num_embeds = core_gguf::kv_i32(meta, "f5.text_num_embeds", hp.text_num_embeds);
    hp.conv_layers = core_gguf::kv_i32(meta, "f5.conv_layers", hp.conv_layers);
    hp.mel_dim = core_gguf::kv_i32(meta, "f5.mel_dim", hp.mel_dim);
    hp.sample_rate = core_gguf::kv_i32(meta, "f5.sample_rate", hp.sample_rate);
    hp.hop_length = core_gguf::kv_i32(meta, "f5.hop_length", hp.hop_length);
    hp.win_length = core_gguf::kv_i32(meta, "f5.win_length", hp.win_length);
    hp.n_fft = core_gguf::kv_i32(meta, "f5.n_fft", hp.n_fft);
    hp.ode_steps = core_gguf::kv_i32(meta, "f5.ode_steps", hp.ode_steps);
    hp.cfg_strength = core_gguf::kv_f32(meta, "f5.cfg_strength", hp.cfg_strength);
    hp.sway_coef = core_gguf::kv_f32(meta, "f5.sway_sampling_coef", hp.sway_coef);
    hp.voc_dim = core_gguf::kv_i32(meta, "f5.voc_dim", hp.voc_dim);
    hp.voc_intermediate_dim = core_gguf::kv_i32(meta, "f5.voc_intermediate_dim", hp.voc_intermediate_dim);
    hp.voc_num_layers = core_gguf::kv_i32(meta, "f5.voc_num_layers", hp.voc_num_layers);
    hp.voc_n_fft = core_gguf::kv_i32(meta, "f5.voc_n_fft", hp.voc_n_fft);
    hp.voc_hop_length = core_gguf::kv_i32(meta, "f5.voc_hop_length", hp.voc_hop_length);

    // Vocab
    auto vocab_chars = core_gguf::kv_str_array(meta, "f5.vocab");
    for (size_t i = 0; i < vocab_chars.size(); i++) {
        ctx->vocab.chars.push_back(vocab_chars[i]);
        ctx->vocab.char_to_idx[vocab_chars[i]] = (int)i;
    }

    core_gguf::free_metadata(meta);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: dim=%d depth=%d heads=%d ff_mult=%d text_dim=%d mel=%d sr=%d\n", hp.dim, hp.depth,
                hp.heads, hp.ff_mult, hp.text_dim, hp.mel_dim, hp.sample_rate);
        fprintf(stderr, "f5_tts: vocab=%zu voc_dim=%d voc_layers=%d\n", ctx->vocab.chars.size(), hp.voc_dim,
                hp.voc_num_layers);
    }

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "f5_tts", wl))
        return false;
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;
    auto& ts = wl.tensors;

    auto get = [&](const char* name) -> ggml_tensor* { return core_gguf::require(ts, name, "f5_tts"); };
    auto try_get = [&](const char* name) -> ggml_tensor* { return core_gguf::try_get(ts, name); };

    // Text embedding
    w.text_emb_weight = get("f5.text_emb.weight");

    // Text ConvNeXtV2 blocks
    w.text_blocks.resize(hp.conv_layers);
    for (int i = 0; i < hp.conv_layers; i++) {
        char buf[128];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "f5.text_blk.%d.%s", i, suffix);
            return get(buf);
        };
        w.text_blocks[i] = {
            g("dw.weight"),  g("dw.bias"),        g("norm.weight"),  g("norm.bias"), g("pw_up.weight"),
            g("pw_up.bias"), g("pw_down.weight"), g("pw_down.bias"), g("grn_gamma"), g("grn_beta"),
        };
    }

    // Time embedding
    w.time_mlp_0_weight = get("f5.time_mlp_0.weight");
    w.time_mlp_0_bias = get("f5.time_mlp_0.bias");
    w.time_mlp_1_weight = get("f5.time_mlp_1.weight");
    w.time_mlp_1_bias = get("f5.time_mlp_1.bias");

    // Input embedding
    w.input_proj_weight = get("f5.input_proj.weight");
    w.input_proj_bias = get("f5.input_proj.bias");
    w.conv_pos_0_weight = get("f5.conv_pos_0.weight");
    w.conv_pos_0_bias = get("f5.conv_pos_0.bias");
    w.conv_pos_1_weight = get("f5.conv_pos_1.weight");
    w.conv_pos_1_bias = get("f5.conv_pos_1.bias");

    // DiT blocks
    w.dit_blocks.resize(hp.depth);
    for (int i = 0; i < hp.depth; i++) {
        char buf[128];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "f5.blk.%d.%s", i, suffix);
            return get(buf);
        };
        w.dit_blocks[i] = {
            g("adaln.weight"),  g("adaln.bias"),    g("attn_q.weight"),   g("attn_q.bias"),   g("attn_k.weight"),
            g("attn_k.bias"),   g("attn_v.weight"), g("attn_v.bias"),     g("attn_o.weight"), g("attn_o.bias"),
            g("ffn_up.weight"), g("ffn_up.bias"),   g("ffn_down.weight"), g("ffn_down.bias"),
        };
    }

    // Final norm + proj
    w.final_adaln_weight = get("f5.final_adaln.weight");
    w.final_adaln_bias = get("f5.final_adaln.bias");
    w.final_proj_weight = get("f5.final_proj.weight");
    w.final_proj_bias = get("f5.final_proj.bias");

    // Rotary
    w.rotary_inv_freq = get("f5.rotary_inv_freq");

    // Vocos
    w.voc_embed_weight = get("voc.embed.weight");
    w.voc_embed_bias = get("voc.embed.bias");
    w.voc_norm_weight = get("voc.norm.weight");
    w.voc_norm_bias = get("voc.norm.bias");
    w.voc_blocks.resize(hp.voc_num_layers);
    for (int i = 0; i < hp.voc_num_layers; i++) {
        char buf[128];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "voc.blk.%d.%s", i, suffix);
            return get(buf);
        };
        w.voc_blocks[i] = {
            g("dw.weight"),  g("dw.bias"),        g("norm.weight"),  g("norm.bias"),   g("pw_up.weight"),
            g("pw_up.bias"), g("pw_down.weight"), g("pw_down.bias"), g("layer_scale"),
        };
    }
    w.voc_final_norm_weight = get("voc.final_norm.weight");
    w.voc_final_norm_bias = get("voc.final_norm.bias");
    w.voc_head_weight = get("voc.head.weight");
    w.voc_head_bias = get("voc.head.bias");

    return true;
}

// ── Public API ───────────────────────────────────────────────────

struct f5_tts_params f5_tts_default_params(void) {
    return {
        /* n_threads     */ 4,
        /* verbosity     */ 1,
        /* use_gpu       */ false,
        /* seed          */ 42,
        /* ode_steps     */ 0,    // 0 = use model default
        /* cfg_strength  */ 0.0f, // 0 = use model default
        /* sway_coef     */ 0.0f, // 0 = use model default
        /* speed         */ 1.0f,
    };
}

struct f5_tts_context* f5_tts_init_from_file(const char* path_model, struct f5_tts_params params) {
    auto* ctx = new f5_tts_context();
    ctx->verbosity = params.verbosity;
    ctx->n_threads = params.n_threads;
    ctx->seed = params.seed;
    ctx->speed = params.speed;

    // Initialize backends
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "f5_tts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (params.verbosity >= 1 && ctx->backend != ctx->backend_cpu) {
        fprintf(stderr, "f5_tts: using GPU backend: %s\n", ggml_backend_name(ctx->backend));
    }

    // Load weights
    if (!load_weights(ctx, path_model)) {
        fprintf(stderr, "f5_tts: failed to load model: %s\n", path_model);
        f5_tts_free(ctx);
        return nullptr;
    }

    // Create backend scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
        if (!ctx->sched) {
            fprintf(stderr, "f5_tts: failed to create backend scheduler\n");
            f5_tts_free(ctx);
            return nullptr;
        }
    }

    // Apply params (0 = use model default)
    ctx->ode_steps = params.ode_steps > 0 ? params.ode_steps : ctx->hp.ode_steps;
    ctx->cfg_strength = params.cfg_strength > 0.0f ? params.cfg_strength : ctx->hp.cfg_strength;
    ctx->sway_coef = params.sway_coef != 0.0f ? params.sway_coef : ctx->hp.sway_coef;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: loaded %s  ode=%d cfg=%.1f sway=%.1f seed=%d\n", path_model, ctx->ode_steps,
                ctx->cfg_strength, ctx->sway_coef, ctx->seed);
    }

    return ctx;
}

void f5_tts_free(struct f5_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->w_buf)
        ggml_backend_buffer_free(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

int f5_tts_set_reference(struct f5_tts_context* ctx, const float* pcm_24k, int n_samples, const char* ref_text) {
    if (!ctx || !pcm_24k || n_samples <= 0)
        return -1;

    // Compute mel spectrogram of reference audio
    int T_ref;
    ctx->ref_mel = compute_mel_spectrogram(pcm_24k, n_samples, ctx->hp.n_fft, ctx->hp.hop_length, ctx->hp.win_length,
                                           ctx->hp.mel_dim, T_ref);

    // If mel computation not yet implemented, allow setting ref_mel directly
    // via the diff harness (which provides it as a GGUF tensor)
    ctx->ref_mel_T = T_ref;
    ctx->ref_text = ref_text ? ref_text : "";
    return 0;
}

int f5_tts_synthesize(struct f5_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    if (!ctx || !text || !pcm_out || !sample_rate_out)
        return 0;

    const auto& hp = ctx->hp;
    int mel_dim = hp.mel_dim;
    int text_dim = hp.text_dim;

    // ── Text preparation ──
    std::string ref_text = ctx->ref_text;
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (!ref_text.empty() && !ends_with(ref_text, ". ") && !ends_with(ref_text, "。")) {
        if (ends_with(ref_text, "."))
            ref_text += " ";
        else
            ref_text += ". ";
    }
    std::string full_text = ref_text + text;

    // Convert to pinyin chars
    auto pinyin_chars = convert_to_pinyin(full_text);
    std::string flat_text;
    for (auto& c : pinyin_chars)
        flat_text += c;

    // Tokenize
    auto tokens = tokenize_text(ctx->vocab, flat_text);

    // ── Duration estimation ──
    // The formula estimates speech rate from (ref_T / ref_text_len) mel frames
    // per character, then applies it to gen_text. When ref_text is empty we
    // have no transcript to derive the rate from, so we estimate the ref
    // transcript length from audio duration at ~13 chars/sec (typical English
    // including spaces — calibrated against F5-TTS upstream defaults).
    int ref_T = ctx->ref_mel_T;
    int ref_text_len;
    if (ref_text.empty()) {
        float mel_fps = (float)ctx->hp.sample_rate / (float)ctx->hp.hop_length;
        float ref_secs = (float)ref_T / mel_fps;
        ref_text_len = std::max(1, (int)(ref_secs * 13.0f));
    } else {
        ref_text_len = (int)ref_text.size();
    }
    int gen_text_len = (int)strlen(text);
    int duration = ref_T + (int)((float)ref_T / (float)ref_text_len * (float)gen_text_len / ctx->speed);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: ref_T=%d duration=%d tokens=%zu text='%s'\n", ref_T, duration, tokens.size(),
                full_text.c_str());
    }

    // ── Text embedding ──
    auto text_emb = compute_text_embed(ctx, tokens.data(), (int)tokens.size(), duration);
    if (text_emb.empty())
        return 0;
    dump_stage(ctx, "text_embed", text_emb.data(), text_emb.size());

    // Unconditional text embedding (all zeros)
    std::vector<float> text_emb_uncond(duration * text_dim, 0.0f);

    // ── Conditioning (ref mel padded to duration) ──
    std::vector<float> cond(duration * mel_dim, 0.0f);
    for (int t = 0; t < ref_T && t < duration; t++) {
        for (int d = 0; d < mel_dim; d++) {
            cond[t * mel_dim + d] = ctx->ref_mel[t * mel_dim + d];
        }
    }
    dump_stage(ctx, "conditioning_input", cond.data(), cond.size());

    // Conditioning mask: where cond != 0 → use cond
    std::vector<float> step_cond(duration * mel_dim, 0.0f);
    for (int t = 0; t < ref_T; t++) {
        for (int d = 0; d < mel_dim; d++) {
            step_cond[t * mel_dim + d] = cond[t * mel_dim + d];
        }
    }

    // ── ODE solve ──
    auto generated = euler_solve(ctx, step_cond, text_emb, text_emb_uncond, duration, mel_dim, text_dim);
    if (generated.empty())
        return 0;

    // ── Extract generated portion (after ref_T) ──
    int gen_T = duration - ref_T;
    if (gen_T <= 0)
        return 0;
    std::vector<float> gen_mel(gen_T * mel_dim);
    for (int t = 0; t < gen_T; t++) {
        for (int d = 0; d < mel_dim; d++) {
            gen_mel[t * mel_dim + d] = generated[(ref_T + t) * mel_dim + d];
        }
    }
    dump_stage(ctx, "vocos_input", gen_mel.data(), gen_mel.size());

    // ── Vocos vocoder ──
    auto audio = vocos_decode(ctx, gen_mel.data(), gen_T, mel_dim);
    if (audio.empty()) {
        // Fallback: return empty for now, will be filled once vocos is implemented
        *pcm_out = nullptr;
        *sample_rate_out = hp.sample_rate;
        return 0;
    }

    // Copy to malloc'd buffer for caller
    int n_samples = (int)audio.size();
    float* out = (float*)malloc(n_samples * sizeof(float));
    memcpy(out, audio.data(), n_samples * sizeof(float));
    *pcm_out = out;
    *sample_rate_out = hp.sample_rate;
    return n_samples;
}

void f5_tts_set_seed(struct f5_tts_context* ctx, int seed) {
    if (ctx)
        ctx->seed = seed;
}
void f5_tts_set_ode_steps(struct f5_tts_context* ctx, int steps) {
    if (ctx)
        ctx->ode_steps = steps;
}
void f5_tts_set_cfg_strength(struct f5_tts_context* ctx, float s) {
    if (ctx)
        ctx->cfg_strength = s;
}
void f5_tts_set_speed(struct f5_tts_context* ctx, float s) {
    if (ctx)
        ctx->speed = s;
}
int f5_tts_sample_rate(const struct f5_tts_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 24000;
}
int f5_tts_vocab_size(const struct f5_tts_context* ctx) {
    return ctx ? (int)ctx->vocab.chars.size() : 0;
}
void f5_tts_set_dump_dir(struct f5_tts_context* ctx, const char* dir) {
    if (ctx)
        ctx->dump_dir = dir ? dir : "";
}

// Test-only: inject reference initial noise for reproducible diff testing.
void f5_tts_set_init_noise(f5_tts_context* ctx, const float* noise, int n) {
    if (!ctx || !noise || n <= 0)
        return;
    ctx->ref_init_noise.assign(noise, noise + n);
}

// Test-only: inject reference mel directly (bypasses mel computation).
void f5_tts_set_ref_mel(f5_tts_context* ctx, const float* mel, int T, int mel_dim, const char* ref_text) {
    if (!ctx)
        return;
    ctx->ref_mel.assign(mel, mel + T * mel_dim);
    ctx->ref_mel_T = T;
    ctx->ref_text = ref_text ? ref_text : "";
}
