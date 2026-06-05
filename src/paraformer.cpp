// src/paraformer.cpp — FunASR Paraformer NAR-ASR runtime.
//
// Inference flow:
//   Kaldi fbank (80-mel) → LFR(7,6) → CMVN → 50 SANM blocks → CIF predictor
//   → 16 decoder blocks (FFN → FSMN → cross-attn) → argmax → characters
//
// The encoder reuses core_sanm::build_block(). See paraformer.h for overview.

#include "paraformer.h"

#include "core/kaldi_fbank.h"
#include "core/lfr.h"
#include "core/sanm.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ===========================================================================
// Model definition
// ===========================================================================

struct paraformer_decoder_block {
    // Upstream order: norm1→FFN, norm2→FSMN, norm3→cross-attn
    // norm1 (before FFN)
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    // FFN: w_1 → relu → internal LN → w_2 (no w_2 bias)
    ggml_tensor *ffn_l1_w = nullptr, *ffn_l1_b = nullptr;
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor* ffn_l2_w = nullptr;
    // norm2 (before FSMN)
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    // FSMN depthwise conv (self-attn substitute)
    ggml_tensor* fsmn_w = nullptr; // (n_feat, K)
    // norm3 (before cross-attention)
    ggml_tensor *norm3_w = nullptr, *norm3_b = nullptr;
    // Cross-attention: Q from decoder, fused K+V from encoder
    ggml_tensor *cross_q_w = nullptr, *cross_q_b = nullptr;
    ggml_tensor *cross_kv_w = nullptr, *cross_kv_b = nullptr;
    ggml_tensor *cross_out_w = nullptr, *cross_out_b = nullptr;
};

struct paraformer_decoder_post {
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *ffn_l1_w = nullptr, *ffn_l1_b = nullptr;
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor* ffn_l2_w = nullptr;
};

struct paraformer_model {
    struct {
        uint32_t n_mels = 80, lfr_m = 7, lfr_n = 6;
        uint32_t sample_rate = 16000;
        uint32_t frame_length_ms = 25, frame_shift_ms = 10;
        uint32_t d_model = 512, n_heads = 4, ffn_dim = 2048;
        uint32_t n_enc_blocks0 = 1, n_enc_blocks = 49, sanm_kernel = 11;
        uint32_t n_dec_blocks = 16, dec_ffn_dim = 2048;
        uint32_t cif_conv_kernel = 3;
        uint32_t vocab_size = 8404;
        float ln_eps = 1e-5f;
    } hp;

    // CMVN
    ggml_tensor *cmvn_shift = nullptr, *cmvn_scale = nullptr;

    // Encoder: entry block(s)
    std::vector<core_sanm::BlockWeights> enc0;
    // Encoder: main blocks
    std::vector<core_sanm::BlockWeights> enc;
    // Encoder: after_norm
    ggml_tensor *enc_after_norm_w = nullptr, *enc_after_norm_b = nullptr;

    // CIF predictor
    ggml_tensor *cif_conv_w = nullptr, *cif_conv_b = nullptr;
    ggml_tensor *cif_out_w = nullptr, *cif_out_b = nullptr;

    // Decoder blocks
    std::vector<paraformer_decoder_block> dec;
    paraformer_decoder_post dec_post;
    ggml_tensor *dec_after_norm_w = nullptr, *dec_after_norm_b = nullptr;
    ggml_tensor *dec_output_w = nullptr, *dec_output_b = nullptr;
    ggml_tensor* dec_embed_w = nullptr; // (vocab, d_model)
};

struct paraformer_context {
    paraformer_model model;
    core_gguf::WeightLoad wl;
    std::string model_path;
    ggml_backend_t backend = nullptr;
    int n_threads = 4;
    bool flash_attn = true;
    int verbosity = 0;
    std::vector<std::string> vocab; // token strings
    std::vector<char> compute_meta;
};

// ===========================================================================
// Helpers
// ===========================================================================

// mm_bias: matmul + optional bias add (shared helper).
static inline ggml_tensor* mm_bias(ggml_context* ctx0, ggml_tensor* W, ggml_tensor* x, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx0, W, x);
    return b ? ggml_add(ctx0, y, b) : y;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool paraformer_load_model(paraformer_context* ctx) {
    auto& m = ctx->model;
    auto& hp = m.hp;
    // Keep a reference to wl for require()
    struct {
        core_gguf::WeightLoad* tensors_ref;
    } wl_wrapper;

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = ctx->wl.tensors.find(name);
        if (it == ctx->wl.tensors.end()) {
            fprintf(stderr, "paraformer: missing tensor '%s'\n", name);
            return nullptr;
        }
        return it->second;
    };

    // Read hyperparameters from GGUF KV
    if (gguf_context* gctx = core_gguf::open_metadata(ctx->model_path.c_str())) {
        hp.d_model = core_gguf::kv_u32(gctx, "paraformer.d_model", hp.d_model);
        hp.n_heads = core_gguf::kv_u32(gctx, "paraformer.n_heads", hp.n_heads);
        hp.n_enc_blocks0 = core_gguf::kv_u32(gctx, "paraformer.n_enc_blocks0", hp.n_enc_blocks0);
        hp.n_enc_blocks = core_gguf::kv_u32(gctx, "paraformer.n_enc_blocks", hp.n_enc_blocks);
        hp.sanm_kernel = core_gguf::kv_u32(gctx, "paraformer.sanm_kernel", hp.sanm_kernel);
        hp.n_dec_blocks = core_gguf::kv_u32(gctx, "paraformer.n_dec_blocks", hp.n_dec_blocks);
        hp.vocab_size = core_gguf::kv_u32(gctx, "paraformer.vocab_size", hp.vocab_size);
        hp.lfr_m = core_gguf::kv_u32(gctx, "paraformer.lfr_m", hp.lfr_m);
        hp.lfr_n = core_gguf::kv_u32(gctx, "paraformer.lfr_n", hp.lfr_n);
        core_gguf::free_metadata(gctx);
    }

    // CMVN
    m.cmvn_shift = get("paraformer.cmvn_shift");
    m.cmvn_scale = get("paraformer.cmvn_scale");

    // Encoder: entry block(s)
    m.enc0.resize(hp.n_enc_blocks0);
    for (uint32_t i = 0; i < hp.n_enc_blocks0; i++) {
        char buf[128];
        auto& b = m.enc0[i];
        auto g = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "paraformer.enc0.blk.%u.%s", i, suf);
            return get(buf);
        };
        b.norm1_w = g("norm1.w");
        b.norm1_b = g("norm1.b");
        b.norm2_w = g("norm2.w");
        b.norm2_b = g("norm2.b");
        b.attn_qkv_w = g("attn.qkv.w");
        b.attn_qkv_b = g("attn.qkv.b");
        b.attn_out_w = g("attn.out.w");
        b.attn_out_b = g("attn.out.b");
        b.attn_fsmn_w = g("attn.fsmn.w");
        b.ffn_l1_w = g("ffn.l1.w");
        b.ffn_l1_b = g("ffn.l1.b");
        b.ffn_l2_w = g("ffn.l2.w");
        b.ffn_l2_b = g("ffn.l2.b");
    }

    // Encoder: main blocks
    m.enc.resize(hp.n_enc_blocks);
    for (uint32_t i = 0; i < hp.n_enc_blocks; i++) {
        char buf[128];
        auto& b = m.enc[i];
        auto g = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "paraformer.enc.blk.%u.%s", i, suf);
            return get(buf);
        };
        b.norm1_w = g("norm1.w");
        b.norm1_b = g("norm1.b");
        b.norm2_w = g("norm2.w");
        b.norm2_b = g("norm2.b");
        b.attn_qkv_w = g("attn.qkv.w");
        b.attn_qkv_b = g("attn.qkv.b");
        b.attn_out_w = g("attn.out.w");
        b.attn_out_b = g("attn.out.b");
        b.attn_fsmn_w = g("attn.fsmn.w");
        b.ffn_l1_w = g("ffn.l1.w");
        b.ffn_l1_b = g("ffn.l1.b");
        b.ffn_l2_w = g("ffn.l2.w");
        b.ffn_l2_b = g("ffn.l2.b");
    }

    m.enc_after_norm_w = get("paraformer.enc.after_norm.w");
    m.enc_after_norm_b = get("paraformer.enc.after_norm.b");

    // CIF predictor
    m.cif_conv_w = get("paraformer.cif.conv.w");
    m.cif_conv_b = get("paraformer.cif.conv.b");
    m.cif_out_w = get("paraformer.cif.out.w");
    m.cif_out_b = get("paraformer.cif.out.b");

    // Decoder blocks
    m.dec.resize(hp.n_dec_blocks);
    for (uint32_t i = 0; i < hp.n_dec_blocks; i++) {
        char buf[128];
        auto& b = m.dec[i];
        auto g = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "paraformer.dec.blk.%u.%s", i, suf);
            return get(buf);
        };
        b.norm1_w = g("norm1.w");
        b.norm1_b = g("norm1.b");
        b.fsmn_w = g("fsmn.w");
        b.norm2_w = g("norm2.w");
        b.norm2_b = g("norm2.b");
        b.cross_q_w = g("cross.q.w");
        b.cross_q_b = g("cross.q.b");
        b.cross_kv_w = g("cross.kv.w");
        b.cross_kv_b = g("cross.kv.b");
        b.cross_out_w = g("cross.out.w");
        b.cross_out_b = g("cross.out.b");
        b.norm3_w = g("norm3.w");
        b.norm3_b = g("norm3.b");
        b.ffn_l1_w = g("ffn.l1.w");
        b.ffn_l1_b = g("ffn.l1.b");
        b.ffn_norm_w = g("ffn.norm.w");
        b.ffn_norm_b = g("ffn.norm.b");
        b.ffn_l2_w = g("ffn.l2.w");
    }

    // Decoder: post block
    {
        auto g = [&](const char* suf) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "paraformer.dec.post.%s", suf);
            return get(buf);
        };
        m.dec_post.norm1_w = g("norm1.w");
        m.dec_post.norm1_b = g("norm1.b");
        m.dec_post.ffn_l1_w = g("ffn.l1.w");
        m.dec_post.ffn_l1_b = g("ffn.l1.b");
        m.dec_post.ffn_norm_w = g("ffn.norm.w");
        m.dec_post.ffn_norm_b = g("ffn.norm.b");
        m.dec_post.ffn_l2_w = g("ffn.l2.w");
    }

    m.dec_after_norm_w = get("paraformer.dec.after_norm.w");
    m.dec_after_norm_b = get("paraformer.dec.after_norm.b");
    m.dec_output_w = get("paraformer.dec.output.w");
    m.dec_output_b = get("paraformer.dec.output.b");
    m.dec_embed_w = get("paraformer.dec.embed.w");

    return true;
}

// ===========================================================================
// Feature extraction (fbank + LFR + CMVN)
// ===========================================================================

static std::vector<float> paraformer_compute_features(paraformer_context* ctx, const float* pcm, int n_samples,
                                                      int& T_lfr, int& D_lfr) {
    const auto& hp = ctx->model.hp;

    // Kaldi fbank
    core_kaldi::FbankParams fp;
    fp.sample_rate = (int)hp.sample_rate;
    fp.n_mels = (int)hp.n_mels;
    fp.frame_length_ms = (int)hp.frame_length_ms;
    fp.frame_shift_ms = (int)hp.frame_shift_ms;
    fp.int16_scale = true;
    fp.window_type = core_kaldi::WindowType::Hamming;

    int T_raw = 0;
    auto fbank = core_kaldi::compute_fbank(pcm, n_samples, fp, T_raw);
    if (T_raw == 0)
        return {};

    // LFR (stack lfr_m frames with stride lfr_n)
    auto lfr = core_lfr::stack(fbank.data(), T_raw, (int)hp.n_mels, (int)hp.lfr_m, (int)hp.lfr_n, T_lfr);
    D_lfr = (int)hp.n_mels * (int)hp.lfr_m; // 80*7 = 560

    // CMVN: x = (x + shift) * scale (AddShift + Rescale)
    if (ctx->model.cmvn_shift && ctx->model.cmvn_scale) {
        const size_t cmvn_sz = ggml_nbytes(ctx->model.cmvn_shift);
        fprintf(stderr, "paraformer: CMVN shift nbytes=%zu, D_lfr=%d, expect=%zu\n", cmvn_sz, D_lfr,
                (size_t)D_lfr * sizeof(float));
        std::vector<float> shift_v((size_t)D_lfr);
        std::vector<float> scale_v((size_t)D_lfr);
        ggml_backend_tensor_get(ctx->model.cmvn_shift, shift_v.data(), 0, D_lfr * sizeof(float));
        ggml_backend_tensor_get(ctx->model.cmvn_scale, scale_v.data(), 0, D_lfr * sizeof(float));
        for (int t = 0; t < T_lfr; t++) {
            for (int d = 0; d < D_lfr; d++) {
                lfr[(size_t)t * D_lfr + d] = (lfr[(size_t)t * D_lfr + d] + shift_v[d]) * scale_v[d];
            }
        }
    }
    return lfr;
}

// ===========================================================================
// CIF predictor (CPU, not a ggml graph — simple sequential loop)
// ===========================================================================

static void cif_predict(paraformer_context* ctx, const float* enc_out, int T, int D,
                        std::vector<float>& acoustic_embeds, int& N_tokens, std::vector<int>* fire_frames = nullptr) {
    // Conv1d(D, D, 3, padding=1) → ReLU → Linear(D, 1) → sigmoid
    // Then CIF accumulation loop.

    // Read weights, dequantizing F16 as needed.
    auto read_f32 = [](ggml_tensor* t, float* dst, size_t n_elem) {
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, dst, 0, n_elem * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<uint16_t> tmp(n_elem);
            ggml_backend_tensor_get(t, tmp.data(), 0, n_elem * sizeof(uint16_t));
            for (size_t i = 0; i < n_elem; i++)
                dst[i] = ggml_fp16_to_fp32((ggml_fp16_t)tmp[i]);
        }
    };

    std::vector<float> conv_w((size_t)D * D * 3);
    std::vector<float> conv_b((size_t)D);
    std::vector<float> out_w((size_t)D);
    float out_b = 0.0f;

    read_f32(ctx->model.cif_conv_w, conv_w.data(), conv_w.size());
    read_f32(ctx->model.cif_conv_b, conv_b.data(), conv_b.size());
    read_f32(ctx->model.cif_out_w, out_w.data(), out_w.size());
    read_f32(ctx->model.cif_out_b, &out_b, 1);

    // Conv1d: for each time step, sum over 3 kernel positions
    // conv_w shape: (out_ch=D, in_ch=D, K=3) → for each output channel o:
    //   y[t, o] = bias[o] + sum_{k=0}^{2} sum_{d=0}^{D-1} w[o, d, k] * x[t-1+k, d]
    // But since this is depthwise-like (in_ch == out_ch == D), we can optimize.
    // Actually it's a full Conv1d(D, D, 3) — not depthwise. Let me do it properly.

    // Apply conv1d with padding=1
    std::vector<float> conv_out((size_t)T * D, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < D; o++) {
            float val = conv_b[o];
            for (int k = 0; k < 3; k++) {
                int ti = t - 1 + k; // padding=1: shift by 1
                if (ti < 0 || ti >= T)
                    continue;
                for (int d = 0; d < D; d++) {
                    // conv_w layout: (out_ch, in_ch, K) → w[o * D * 3 + d * 3 + k]
                    val += conv_w[(size_t)o * D * 3 + (size_t)d * 3 + k] * enc_out[(size_t)ti * D + d];
                }
            }
            conv_out[(size_t)t * D + o] = val > 0.0f ? val : 0.0f; // ReLU
        }
    }

    // Linear(D, 1) → sigmoid → alphas
    std::vector<float> alphas((size_t)T);
    for (int t = 0; t < T; t++) {
        float val = out_b;
        for (int d = 0; d < D; d++) {
            val += out_w[d] * conv_out[(size_t)t * D + d];
        }
        alphas[t] = 1.0f / (1.0f + std::exp(-val)); // sigmoid
    }

    // CIF accumulation: integrate alphas, fire when ≥ 1.0
    const float tail_threshold = 0.45f;
    float accum = 0.0f;
    std::vector<float> fired; // (D,) vectors, one per fired token
    std::vector<float> partial((size_t)D, 0.0f);

    for (int t = 0; t < T; t++) {
        float alpha = alphas[t];
        accum += alpha;
        if (accum >= 1.0f) {
            // Fire: blend current frame
            float remainder = 1.0f - (accum - alpha);
            for (int d = 0; d < D; d++) {
                partial[d] += remainder * enc_out[(size_t)t * D + d];
            }
            fired.insert(fired.end(), partial.begin(), partial.end());
            if (fire_frames)
                fire_frames->push_back(t);
            // Start new token with leftover
            float leftover = accum - 1.0f;
            for (int d = 0; d < D; d++) {
                partial[d] = leftover * enc_out[(size_t)t * D + d];
            }
            accum = leftover;
        } else {
            for (int d = 0; d < D; d++) {
                partial[d] += alpha * enc_out[(size_t)t * D + d];
            }
        }
    }
    // Tail handling: if remaining accum > threshold, fire one more
    if (accum > tail_threshold) {
        fired.insert(fired.end(), partial.begin(), partial.end());
        if (fire_frames)
            fire_frames->push_back(T - 1);
    }

    N_tokens = (int)(fired.size() / (size_t)D);
    acoustic_embeds = std::move(fired);
}

// ===========================================================================
// Decoder graph build (FSMN + cross-attention + FFN-with-LN)
// ===========================================================================

static ggml_tensor* build_decoder_block(ggml_context* ctx0, ggml_tensor* cur, ggml_tensor* enc_out, int N, int T_enc,
                                        const paraformer_decoder_block& b, int D, int n_heads, int head_dim, float eps,
                                        bool flash_attn) {
    // Upstream order (DecoderLayerSANM.forward):
    //   1. norm1 → FFN (no residual yet)
    //   2. norm2 → FSMN (with internal conv+identity residual) → add original input
    //   3. norm3 → cross-attn → add residual

    // 1. FFN: norm1 → w_1 → relu → internal LN → w_2
    ggml_tensor* residual = cur;
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, b.norm1_w, b.norm1_b, eps);
    x = mm_bias(ctx0, b.ffn_l1_w, x, b.ffn_l1_b); // (ffn_dim, N)
    x = ggml_relu(ctx0, x);
    x = ggml_norm_affine(ctx0, x, b.ffn_norm_w, b.ffn_norm_b, eps); // internal LN after activation
    x = ggml_mul_mat(ctx0, b.ffn_l2_w, x);                          // no bias on w_2
    // No residual addition here — residual spans FFN + FSMN together.

    // 2. FSMN self-attention: norm2 → depthwise conv + internal residual → add original input
    ggml_tensor* ffn_out = x;
    {
        ggml_tensor* normed = ggml_norm_affine(ctx0, ffn_out, b.norm2_w, b.norm2_b, eps);

        const int K = (int)b.fsmn_w->ne[0]; // kernel width; ne = (K, D)
        ggml_tensor* w4 = ggml_cast(ctx0, b.fsmn_w, GGML_TYPE_F32);
        w4 = ggml_reshape_4d(ctx0, w4, K, 1, 1, D);

        ggml_tensor* v4 = ggml_cont(ctx0, ggml_transpose(ctx0, normed)); // (N, D)
        v4 = ggml_reshape_4d(ctx0, v4, N, 1, D, 1);
        ggml_tensor* y = ggml_conv_2d_dw_direct(ctx0, w4, v4, 1, 1, (K - 1) / 2, 0, 1, 1);
        y = ggml_cont(ctx0, ggml_permute(ctx0, y, 1, 2, 0, 3));
        y = ggml_reshape_2d(ctx0, y, D, N);
        x = ggml_add(ctx0, y, normed); // FSMN internal residual: conv(normed) + normed
    }
    cur = ggml_add(ctx0, residual, x); // add original input residual

    // 3. Cross-attention: norm3 → Q from decoder, K+V from encoder
    residual = cur;
    x = ggml_norm_affine(ctx0, cur, b.norm3_w, b.norm3_b, eps);

    {
        // Q = linear_q(decoder_hidden): (D, N) → (D, N)
        ggml_tensor* Q = mm_bias(ctx0, b.cross_q_w, x, b.cross_q_b);

        // K, V = split(linear_k_v(encoder_out), D): (2D, T_enc) → K (D, T), V (D, T)
        ggml_tensor* kv = mm_bias(ctx0, b.cross_kv_w, enc_out, b.cross_kv_b); // (2D, T_enc)
        const size_t row_bytes = kv->nb[1];
        ggml_tensor* K_ = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, D, T_enc, row_bytes, 0));
        ggml_tensor* V = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, D, T_enc, row_bytes, (size_t)D * sizeof(float)));

        // Reshape for multi-head attention
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, N);
        K_ = ggml_reshape_3d(ctx0, K_, head_dim, n_heads, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));   // (hd, N, n_heads)
        K_ = ggml_cont(ctx0, ggml_permute(ctx0, K_, 0, 2, 1, 3)); // (hd, T_enc, n_heads)
        ggml_tensor* V_h = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        const float scale = 1.0f / std::sqrt((float)head_dim);

        ggml_tensor* attn;
        if (flash_attn) {
            attn = ggml_flash_attn_ext(ctx0, Q, K_, V_h, nullptr, scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, D, N);
        } else {
            ggml_tensor* scores = ggml_mul_mat(ctx0, K_, Q); // (T_enc, N, n_heads)
            scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);
            ggml_tensor* V_p = ggml_cont(ctx0, ggml_permute(ctx0, V_h, 1, 0, 2, 3));
            attn = ggml_mul_mat(ctx0, V_p, scores);
            attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(ctx0, attn, D, N);
        }
        x = mm_bias(ctx0, b.cross_out_w, attn, b.cross_out_b);
    }
    cur = ggml_add(ctx0, residual, x);

    return cur;
}

// Post-processing block (decoders3): norm1 → w_1 → relu → LN → w_2
// Upstream: self_attn=None, src_attn=None, so no residual addition.
static ggml_tensor* build_decoder_post(ggml_context* ctx0, ggml_tensor* cur, const paraformer_decoder_post& b,
                                       float eps) {
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, b.norm1_w, b.norm1_b, eps);
    x = mm_bias(ctx0, b.ffn_l1_w, x, b.ffn_l1_b);
    x = ggml_relu(ctx0, x);
    x = ggml_norm_affine(ctx0, x, b.ffn_norm_w, b.ffn_norm_b, eps); // LN after activation
    x = ggml_mul_mat(ctx0, b.ffn_l2_w, x);
    return x; // no residual — upstream decoders3 has no self_attn so residual is never added
}

// ===========================================================================
// Full inference
// ===========================================================================

static std::string paraformer_transcribe_impl(paraformer_context* ctx, const float* pcm, int n_samples,
                                              std::vector<float>* staged_out = nullptr, const char* stage = nullptr,
                                              std::vector<int>* out_fire_frames = nullptr) {
    const auto& hp = ctx->model.hp;
    const int D = (int)hp.d_model;

    // 1. Feature extraction
    int T_lfr = 0, D_lfr_actual = 0;
    auto lfr = paraformer_compute_features(ctx, pcm, n_samples, T_lfr, D_lfr_actual);
    if (lfr.empty())
        return "";

    // 2. Build and run encoder graph
    const size_t meta_sz = 256 * 1024 * 1024; // 256 MB compute buffer
    if (ctx->compute_meta.empty())
        ctx->compute_meta.resize(meta_sz);

    ggml_init_params ip = {meta_sz, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input tensor
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D_lfr_actual, T_lfr);
    ggml_set_name(inp, "features");
    ggml_set_input(inp);

    ggml_tensor* cur = inp;

    // Encoder: entry block(s)
    int enc_layer_idx = 0;
    for (uint32_t i = 0; i < hp.n_enc_blocks0; i++) {
        core_sanm::BlockParams p;
        p.in_size = (i == 0) ? D_lfr_actual : D;
        p.size = D;
        p.n_heads = (int)hp.n_heads;
        p.head_dim = D / (int)hp.n_heads;
        p.kernel = (int)hp.sanm_kernel;
        p.ln_eps = hp.ln_eps;
        p.flash_attn = ctx->flash_attn;
        cur = core_sanm::build_block(ctx0, cur, T_lfr, ctx->model.enc0[i], p, /*apply_attn_residual=*/false);
        if (stage) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "encoder_layer_%d", enc_layer_idx);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
        }
        enc_layer_idx++;
    }

    // Encoder: main blocks
    for (uint32_t i = 0; i < hp.n_enc_blocks; i++) {
        core_sanm::BlockParams p;
        p.in_size = D;
        p.size = D;
        p.n_heads = (int)hp.n_heads;
        p.head_dim = D / (int)hp.n_heads;
        p.kernel = (int)hp.sanm_kernel;
        p.ln_eps = hp.ln_eps;
        p.flash_attn = ctx->flash_attn;
        cur = core_sanm::build_block(ctx0, cur, T_lfr, ctx->model.enc[i], p, /*apply_attn_residual=*/true);
        if (stage) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "encoder_layer_%d", enc_layer_idx);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
        }
        enc_layer_idx++;
    }

    // Encoder: after_norm
    cur = ggml_norm_affine(ctx0, cur, ctx->model.enc_after_norm_w, ctx->model.enc_after_norm_b, hp.ln_eps);
    // Mark the output so the allocator doesn't reuse the buffer.
    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate and run encoder
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    {
        const size_t inp_bytes = ggml_nbytes(inp);
        const size_t lfr_bytes = lfr.size() * sizeof(float);
        if (lfr_bytes > inp_bytes) {
            fprintf(stderr, "paraformer: input tensor too small: %zu bytes vs %zu lfr bytes (T=%d, D=%d)\n", inp_bytes,
                    lfr_bytes, T_lfr, D_lfr_actual);
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return "";
        }
        fprintf(stderr, "paraformer: encoder input OK: T_lfr=%d, D_lfr=%d, bytes=%zu\n", T_lfr, D_lfr_actual,
                lfr_bytes);
        ggml_backend_tensor_set(inp, lfr.data(), 0, lfr_bytes);
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    fprintf(stderr, "paraformer: running encoder graph...\n");
    ggml_backend_graph_compute(ctx->backend, gf);
    fprintf(stderr, "paraformer: encoder done\n");

    // Extract encoder output
    ggml_tensor* enc_result = ggml_graph_get_tensor(gf, "encoder_output");
    if (!enc_result) {
        fprintf(stderr, "paraformer: encoder_output tensor not found in graph\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
        return "";
    }
    const size_t enc_bytes = ggml_nbytes(enc_result);
    const size_t enc_expect = (size_t)D * T_lfr * sizeof(float);
    fprintf(stderr, "paraformer: enc_result ne=(%lld, %lld), type=%d, nbytes=%zu, expect=%zu\n",
            (long long)enc_result->ne[0], (long long)enc_result->ne[1], (int)enc_result->type, enc_bytes, enc_expect);
    if (enc_bytes < enc_expect) {
        fprintf(stderr, "paraformer: encoder_output size mismatch: got %zu, expected %zu (D=%d, T=%d)\n", enc_bytes,
                enc_expect, D, T_lfr);
        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
        return "";
    }
    std::vector<float> enc_out((size_t)D * T_lfr);
    ggml_backend_tensor_get(enc_result, enc_out.data(), 0, enc_expect);

    // Stage capture from encoder graph
    if (stage && staged_out) {
        std::string sn(stage);
        if (sn == "mel_features") {
            // lfr is the post-LFR feature vector (T_lfr, D_lfr=560) in row-major
            *staged_out = lfr;
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return "";
        }
        if (sn == "encoder_output") {
            *staged_out = enc_out;
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return "";
        }
        if (sn.rfind("encoder_layer_", 0) == 0) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, stage);
            if (t) {
                staged_out->resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, staged_out->data(), 0, ggml_nbytes(t));
            }
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return "";
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx0);

    // 3. CIF predictor (CPU)
    // enc_out flat storage: ggml tensor ne=(D, T_lfr), element (d,t) at
    // flat index t*D + d.  This IS row-major (T, D), so no transpose needed.
    std::vector<float> acoustic_embeds;
    int N_tokens = 0;
    std::vector<int> fire_frames_vec;
    cif_predict(ctx, enc_out.data(), T_lfr, D, acoustic_embeds, N_tokens, out_fire_frames ? &fire_frames_vec : nullptr);
    if (out_fire_frames)
        *out_fire_frames = std::move(fire_frames_vec);

    fprintf(stderr, "paraformer: CIF predicted %d tokens from T_lfr=%d\n", N_tokens, T_lfr);
    if (N_tokens <= 0)
        return "";

    // Stage capture: acoustic_embeds (row-major (N, D))
    if (stage && staged_out && std::strcmp(stage, "acoustic_embeds") == 0) {
        *staged_out = acoustic_embeds;
        return "";
    }

    // 4. Decoder graph
    ctx0 = ggml_init(ip);
    gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Decoder inputs
    ggml_tensor* dec_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, N_tokens);
    ggml_set_name(dec_inp, "acoustic_embeds");
    ggml_set_input(dec_inp);

    ggml_tensor* enc_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_lfr);
    ggml_set_name(enc_tensor, "enc_for_decoder");
    ggml_set_input(enc_tensor);

    cur = dec_inp;
    const int n_heads = (int)hp.n_heads;
    const int head_dim = D / n_heads;

    for (uint32_t i = 0; i < hp.n_dec_blocks; i++) {
        cur = build_decoder_block(ctx0, cur, enc_tensor, N_tokens, T_lfr, ctx->model.dec[i], D, n_heads, head_dim,
                                  hp.ln_eps, ctx->flash_attn);
        if (stage) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "decoder_layer_%u", i);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
        }
    }

    // Post block
    if (ctx->model.dec_post.norm1_w) {
        cur = build_decoder_post(ctx0, cur, ctx->model.dec_post, hp.ln_eps);
    }

    // after_norm → output_layer
    cur = ggml_norm_affine(ctx0, cur, ctx->model.dec_after_norm_w, ctx->model.dec_after_norm_b, hp.ln_eps);
    if (stage) {
        ggml_set_name(cur, "decoder_output");
        ggml_set_output(cur);
    }
    cur = mm_bias(ctx0, ctx->model.dec_output_w, cur, ctx->model.dec_output_b); // (vocab, N)
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    // Allocate and run decoder
    alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    // Set decoder inputs: acoustic_embeds from CIF is row-major (N, D).
    // ggml tensor ne=(D, N) stores element (d,n) at flat index n*D+d, which
    // is the same layout. No transposition needed.
    ggml_backend_tensor_set(dec_inp, acoustic_embeds.data(), 0, (size_t)D * N_tokens * sizeof(float));
    ggml_backend_tensor_set(enc_tensor, enc_out.data(), 0, enc_out.size() * sizeof(float));
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    ggml_backend_graph_compute(ctx->backend, gf);

    // Stage capture from decoder graph
    if (stage && staged_out) {
        std::string sn(stage);
        if (sn == "decoder_output" || sn.rfind("decoder_layer_", 0) == 0) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, stage);
            if (t) {
                staged_out->resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, staged_out->data(), 0, ggml_nbytes(t));
            }
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return "";
        }
    }

    // 5. Argmax decoding
    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    const int V = (int)hp.vocab_size;
    std::vector<float> logits_data((size_t)V * N_tokens);
    ggml_backend_tensor_get(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));

    std::string result;
    bool prev_was_bpe_cont = false; // true if previous token ended with @@
    for (int n = 0; n < N_tokens; n++) {
        int best = 0;
        float best_val = -1e30f;
        for (int v = 0; v < V; v++) {
            float val = logits_data[(size_t)n * V + v];
            if (val > best_val) {
                best_val = val;
                best = v;
            }
        }
        // Skip special tokens: 0=<blank>, 1=<s>, 2=</s>, last=<unk>
        if (best <= 2 || best == V - 1)
            continue;
        if ((size_t)best < ctx->vocab.size()) {
            std::string tok = ctx->vocab[best];
            bool is_bpe_cont = (tok.size() >= 2 && tok.substr(tok.size() - 2) == "@@");
            // Insert space before Latin-script tokens that start a new word.
            // Paraformer-zh uses character-level Chinese + word-level English
            // with @@ BPE continuation markers. Space is needed between
            // consecutive English words (tokens without @@ on either side).
            if (!result.empty() && !prev_was_bpe_cont) {
                unsigned char first_byte = (unsigned char)tok[0];
                unsigned char last_byte = (unsigned char)result.back();
                bool cur_is_latin =
                    (first_byte >= 'a' && first_byte <= 'z') || (first_byte >= 'A' && first_byte <= 'Z');
                bool prev_is_latin = (last_byte >= 'a' && last_byte <= 'z') || (last_byte >= 'A' && last_byte <= 'Z');
                if (cur_is_latin && prev_is_latin) {
                    result += ' ';
                }
            }
            if (is_bpe_cont) {
                result += tok.substr(0, tok.size() - 2);
            } else {
                result += tok;
            }
            prev_was_bpe_cont = is_bpe_cont;
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx0);

    return result;
}

// ===========================================================================
// Public API
// ===========================================================================

paraformer_context_params paraformer_context_default_params() {
    return {4, 0, true};
}

paraformer_context* paraformer_init_from_file(const char* path, paraformer_context_params params) {
    auto* ctx = new paraformer_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->flash_attn = params.flash_attn;
    ctx->verbosity = params.verbosity;

    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "paraformer: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }

    ctx->model_path = path;
    if (!core_gguf::load_weights(path, ctx->backend, "paraformer", ctx->wl)) {
        fprintf(stderr, "paraformer: failed to load '%s'\n", path);
        delete ctx;
        return nullptr;
    }

    if (!paraformer_load_model(ctx)) {
        delete ctx;
        return nullptr;
    }

    // Load vocabulary from GGUF KV (tokenizer.tokens)
    if (gguf_context* gctx = core_gguf::open_metadata(ctx->model_path.c_str())) {
        int idx = gguf_find_key(gctx, "tokenizer.tokens");
        if (idx >= 0) {
            int n = gguf_get_arr_n(gctx, idx);
            ctx->vocab.resize(n);
            for (int i = 0; i < n; i++) {
                ctx->vocab[i] = gguf_get_arr_str(gctx, idx, i);
            }
        }
        core_gguf::free_metadata(gctx);
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "paraformer: loaded %zu tensors, vocab %zu, enc %u+%u blocks, dec %u blocks\n",
                ctx->wl.tensors.size(), ctx->vocab.size(), ctx->model.hp.n_enc_blocks0, ctx->model.hp.n_enc_blocks,
                ctx->model.hp.n_dec_blocks);
    }

    return ctx;
}

void paraformer_free(paraformer_context* ctx) {
    if (!ctx)
        return;
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

char* paraformer_transcribe(paraformer_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    std::string s = paraformer_transcribe_impl(ctx, samples, n_samples);
    char* buf = (char*)std::malloc(s.size() + 1);
    if (buf) {
        std::memcpy(buf, s.data(), s.size());
        buf[s.size()] = '\0';
    }
    return buf;
}

paraformer_result* paraformer_transcribe_with_timestamps(paraformer_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    std::vector<int> fire_frames;
    std::string s = paraformer_transcribe_impl(ctx, samples, n_samples, nullptr, nullptr, &fire_frames);

    auto* r = (paraformer_result*)std::malloc(sizeof(paraformer_result));
    if (!r)
        return nullptr;
    r->text = (char*)std::malloc(s.size() + 1);
    if (r->text) {
        std::memcpy(r->text, s.data(), s.size());
        r->text[s.size()] = '\0';
    }

    // Convert fire frame indices to centiseconds.
    // LFR (low frame rate) subsampling: each LFR frame covers `lfr_m` raw
    // mel frames. Raw mel frames are 10ms apart (hop=160 at 16kHz).
    // So frame t in the encoder corresponds to t * lfr_m * 10ms.
    const int lfr_m = (int)ctx->model.hp.lfr_m;
    const int ms_per_frame = lfr_m * 10;
    r->n_chars = (int)fire_frames.size();
    r->char_times_cs = (int32_t*)std::malloc(fire_frames.size() * sizeof(int32_t));
    if (r->char_times_cs) {
        for (int i = 0; i < r->n_chars; i++)
            r->char_times_cs[i] = fire_frames[i] * ms_per_frame / 10; // ms → cs
    }
    return r;
}

void paraformer_result_free(paraformer_result* r) {
    if (!r)
        return;
    std::free(r->text);
    std::free(r->char_times_cs);
    std::free(r);
}

float* paraformer_extract_stage(paraformer_context* ctx, const float* samples, int n_samples, const char* stage_name,
                                int* n_out) {
    if (n_out)
        *n_out = 0;
    if (!ctx || !samples || n_samples <= 0 || !stage_name)
        return nullptr;

    if (std::strcmp(stage_name, "generated_text") == 0) {
        std::string txt = paraformer_transcribe_impl(ctx, samples, n_samples);
        char* buf = (char*)std::malloc(txt.size() + 1);
        if (!buf)
            return nullptr;
        std::memcpy(buf, txt.data(), txt.size());
        buf[txt.size()] = '\0';
        if (n_out)
            *n_out = (int)txt.size();
        // cppcheck-suppress invalidPointerCast
        return (float*)buf; // caller casts back to char* (generated_text path)
    }

    std::vector<float> staged;
    (void)paraformer_transcribe_impl(ctx, samples, n_samples, &staged, stage_name);
    if (staged.empty())
        return nullptr;
    float* out = (float*)std::malloc(staged.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, staged.data(), staged.size() * sizeof(float));
    if (n_out)
        *n_out = (int)staged.size();
    return out;
}
