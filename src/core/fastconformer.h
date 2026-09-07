// src/core/fastconformer.h — shared FastConformer encoder helpers.
//
// Replaces the dw_striding pre-encode + rel-pos sinusoidal table + 24-32×
// macaron Conformer block body that parakeet, canary, and canary_ctc each
// have a near-identical copy of. The only per-model difference is whether
// Q/K/V/output projections, FFN linears, and pointwise convs carry biases
// — parakeet and canary_ctc don't, canary does.
//
// The helper is header-only so the compiler inlines it straight into each
// caller, producing the exact same ggml op sequence as the original inline
// code and preserving bit-identical graph execution on the regression sweep.
//
// Scope:
//   core_conformer::rel_shift        — (T-1)-shift view used by rel-pos attn
//   core_conformer::make_pos_enc     — sinusoidal rel-pos table builder
//   core_conformer::PreEncodeWeights — dw_striding subsampling weights
//   core_conformer::BlockWeights     — one Conformer block's tensors
//   core_conformer::build_pre_encode — conv front-end + linear → (d, T)
//   core_conformer::build_block      — one Conformer block
//
// Each model still owns its own ggml_context setup, input tensor creation,
// and final output head (CTC / RNN-T joint / transformer decoder). What
// moves into here is just the shared encoder body.

#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "core/ggml_cpu_backend.h"

namespace core_conformer {

// Env-var gate: CRISPASR_FC_NO_FLASH=1 disables flash_attn_ext in the
// FastConformer encoder and uses manual QK^T + softmax + V instead.
// Useful for A/B-ing the flash path on CPU for short sequences.
static inline bool fc_no_flash() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_FC_NO_FLASH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Env-var gate: CRISPASR_FC_ATTN_CONT=1 restores the legacy ggml_cont copies
// of Q/K/V before flash_attn_ext. The kernel reads strided views directly
// (same as llama.cpp's permuted Q), so the copies are pure overhead — this
// gate exists only for regression bisection (issue #81).
static inline bool fc_attn_cont() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_FC_ATTN_CONT");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// GPU manual-attention gate (issue #81 GPU phase). CUDA's flash_attn_ext
// rejects the FastConformer per-head rel-pos mask (fattn.cu guard on
// mask->ne[2] != 1), so every flash node falls back to CPU — 24 GPU↔CPU
// bounces per encoder pass. The manual QK^T + soft_max_ext + V path runs
// fully on-device there. Metal handles the per-head mask and measured
// FASTER with flash (PERFORMANCE.md 2026-05-09), so auto never fires there.
// Kaggle P100 A/B (2026-07-12, parakeet-ctc q8_0, transcripts identical):
// jfk 11 s 0.180→0.095 s, jfk 55 s 1.140→0.360 s (3.2x) → default ON for
// CUDA. CRISPASR_FC_GPU_MANUAL_ATTN: "0" never, "1" any non-CPU backend,
// unset = auto (CUDA only).
static inline bool fc_gpu_manual_attn(ggml_backend_t backend) {
    static int v = -2;
    if (v == -2) {
        const char* e = std::getenv("CRISPASR_FC_GPU_MANUAL_ATTN");
        v = (!e || !*e) ? -1 : (*e != '0' ? 1 : 0);
    }
    if (v == 0 || !backend || core_cpu_backend::is_cpu(backend))
        return false;
    if (v == 1)
        return true;
    const char* name = ggml_backend_name(backend);
    return name && (name[0] == 'C' || name[0] == 'c') && (name[1] == 'U' || name[1] == 'u');
}

// ---------------------------------------------------------------------------
// Rel-pos "shift" trick from Transformer-XL / Conformer: rewrites the raw
// (Q @ R^T) matrix — which is indexed by (rel_pos, query_pos) with rel_pos
// running over 2T-1 positions — into a square (key_pos, query_pos) matrix
// of size (T, T). This is done as a strided view, no copy.
// Input:  [2T-1, T, H]
// Output: [T,   T, H]
// ---------------------------------------------------------------------------
static inline ggml_tensor* rel_shift(ggml_context* ctx, ggml_tensor* a) {
    const int T = (int)a->ne[1];
    const int H = (int)a->ne[2];
    return ggml_view_3d(ctx, a, T, T, H, a->nb[1] - a->nb[0], a->nb[2], (T - 1) * a->nb[0]);
}

// ---------------------------------------------------------------------------
// Sinusoidal rel-pos table, layout (d_model, 2T-1), with positions running
// descending from +(T-1) to -(T-1). Memory layout is pe[dim + pos*d] so
// that ne[0]=d (fast axis) and ne[1]=2T-1 (slow axis) matches the ggml
// tensor created as ggml_new_tensor_2d(F32, d, 2T-1).
//
// IMPORTANT: the tensor is created as ggml_new_tensor_2d(F32, d, 2T-1),
// giving ne[0]=d (fast) and ne[1]=2T-1 (slow). The CORRECT memory layout
// is therefore `pe[dim + pos*d]`, NOT `pe[(2*i)*K + j]` (which transposes
// the axes). An earlier version of parakeet/cohere shipped with the
// transposed layout — parakeet's TDT decoder is robust enough to mostly
// recover, but canary's encoder–decoder cross-attention is not. If you
// see word boundaries drifting on a new consumer, re-check this first.
// ---------------------------------------------------------------------------
static inline std::vector<float> make_pos_enc(int d_model, int T) {
    const int n_pos = 2 * T - 1;
    std::vector<float> pe((size_t)n_pos * d_model, 0.0f);
    for (int p = 0; p < n_pos; p++) {
        const float pos = (float)(T - 1 - p);
        for (int i = 0; i < d_model / 2; i++) {
            const float div = expf(-logf(10000.0f) * (float)(2 * i) / (float)d_model);
            pe[(size_t)p * d_model + 2 * i] = sinf(pos * div);
            pe[(size_t)p * d_model + 2 * i + 1] = cosf(pos * div);
        }
    }
    return pe;
}

// ---------------------------------------------------------------------------
// Local attention mask builder for rel_pos_local_attn models.
// Returns a flat row-major (T, T) float array where mask[q*T+k] = 0.0 for
// visible positions and -inf for masked positions. Positions within
// [q - left, q + right] are visible. The first `global_tokens` positions
// are always visible to all queries (and can see all positions).
// ---------------------------------------------------------------------------
static inline std::vector<float> make_local_attn_mask(int T, int left, int right, int global_tokens) {
    const float NEG_INF = -1e9f; // large enough for softmax to zero out
    std::vector<float> mask((size_t)T * T, NEG_INF);
    for (int q = 0; q < T; q++) {
        // Global token queries can attend to everything.
        if (q < global_tokens) {
            for (int k = 0; k < T; k++)
                mask[(size_t)q * T + k] = 0.0f;
            continue;
        }
        // Regular positions attend to their local window + global tokens.
        int k_lo = q - left;
        if (k_lo < 0)
            k_lo = 0;
        int k_hi = q + right;
        if (k_hi >= T)
            k_hi = T - 1;
        for (int k = k_lo; k <= k_hi; k++)
            mask[(size_t)q * T + k] = 0.0f;
        // Global token keys are always visible.
        for (int k = 0; k < global_tokens && k < T; k++)
            mask[(size_t)q * T + k] = 0.0f;
    }
    return mask;
}

// Band mask for the TRUE windowed (block sliding-chunks) attention path.
// Layout matches build_windowed_attn: a flat (3*BS, BS, NB) F32 array indexed
// [b*BS+i][3*BS] + j, natural key order — global query q=b*BS+i, global key
// k=(b-1)*BS+j. 0.0 where k is in [0,T) and within [q-left, q+right] (or a
// global token), -inf otherwise. NB = ceil(T/BS). Must use the same BS as
// fc_window_block_size(left, right).
static inline std::vector<float> make_window_band_mask(int T, int left, int right, int global_tokens, int BS) {
    const float NEG_INF = -1e9f;
    const int NB = (T + BS - 1) / BS;
    std::vector<float> mask((size_t)3 * BS * BS * NB, NEG_INF);
    for (int b = 0; b < NB; b++) {
        for (int i = 0; i < BS; i++) {
            const int q = b * BS + i;
            for (int j = 0; j < 3 * BS; j++) {
                const int k = (b - 1) * BS + j;
                bool vis =
                    k >= 0 && k < T && ((k >= q - left && k <= q + right) || k < global_tokens || q < global_tokens);
                mask[((size_t)b * BS + i) * (3 * BS) + j] = vis ? 0.0f : NEG_INF;
            }
        }
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Pre-encode (dw_striding 8× subsampling) weights.
//
//   Conv2d(1→C,  k=3, s=2, p=1) → ReLU
//   Conv2d_dw(C, k=3, s=2, p=1)
//   Conv2d(C→C,  k=1)            → ReLU
//   Conv2d_dw(C, k=3, s=2, p=1)
//   Conv2d(C→C,  k=1)            → ReLU
//   flatten(freq×channel) → Linear(W3*C → d_model)
// ---------------------------------------------------------------------------
struct PreEncodeWeights {
    ggml_tensor *conv0_w = nullptr, *conv0_b = nullptr; // first strided conv
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr; // dw
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr; // pw
    ggml_tensor *conv5_w = nullptr, *conv5_b = nullptr; // dw
    ggml_tensor *conv6_w = nullptr, *conv6_b = nullptr; // pw
    ggml_tensor *out_w = nullptr, *out_b = nullptr;     // Linear(W3*C → d_model)
};

// Snap a 4D conv output (OW, OH, OC, N) as a 2D named dup (OC*OW, OH) for
// staged comparison.  Feature ordering: k = oc*(OW) + ow  (matches Python's
// x.transpose(1,2).reshape(T, C*Freq) convention).
static inline void snap_conv4d(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* t, const char* name) {
    // permute(1, 2, 0, 3): (OW,OH,OC,N) → (OH,OC,OW,N)
    ggml_tensor* p = ggml_cont(ctx0, ggml_permute(ctx0, t, 1, 2, 0, 3));
    // reshape to (OC*OW, OH): ne[0]=OC*OW fastest, ne[1]=OH (T_enc)
    const int64_t C_Freq = t->ne[2] * t->ne[0]; // OC * OW
    const int64_t T_enc = t->ne[1];             // OH
    ggml_tensor* flat = ggml_reshape_2d(ctx0, p, C_Freq, T_enc);
    ggml_tensor* snap = ggml_dup(ctx0, flat);
    ggml_set_name(snap, name);
    ggml_build_forward_expand(gf, snap);
}

// Build the dw_striding pre-encoder. Input `mel` has shape (n_mels, T_mel).
// Returns a (d_model, T_enc) tensor where T_enc is read off the intermediate
// conv output via the caller (write it back through `out_T_enc`).
// When `gf` is non-null, named dup snaps are added after each conv step for
// staged comparison via the diff harness.
// `stage_mask_t0` / `stage_mask_t1` are optional (1, T, 1, 1) multiplicative
// time masks for the bucketed-padding path: zeroing pad frames after the
// stage feeding each k=3 conv makes the padded graph's valid outputs
// bit-identical to the unpadded graph (whose convs zero-pad those reads).
static inline ggml_tensor* build_pre_encode(ggml_context* ctx0, ggml_tensor* mel, const PreEncodeWeights& w,
                                            int subsampling_channels, int* out_T_enc, ggml_cgraph* gf = nullptr,
                                            ggml_tensor* stage_mask_t0 = nullptr,
                                            ggml_tensor* stage_mask_t1 = nullptr) {
    auto bias_4d = [&](ggml_tensor* b) {
        return ggml_cast(ctx0, ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    ggml_tensor* cur = ggml_conv_2d(ctx0, w.conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv0_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c0");
    cur = ggml_relu(ctx0, cur);
    if (stage_mask_t0) // zero pad frames before the next k=3 conv (see header note)
        cur = ggml_mul(ctx0, cur, stage_mask_t0);

    cur = ggml_conv_2d_dw(ctx0, w.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv2_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c2");
    cur = ggml_conv_2d(ctx0, w.conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv3_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c3");
    cur = ggml_relu(ctx0, cur);
    if (stage_mask_t1) // zero pad frames before the next k=3 conv
        cur = ggml_mul(ctx0, cur, stage_mask_t1);

    cur = ggml_conv_2d_dw(ctx0, w.conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv5_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c5");
    cur = ggml_conv_2d(ctx0, w.conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv6_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c6");
    cur = ggml_relu(ctx0, cur);

    const int H3 = (int)cur->ne[1];
    const int W3 = (int)cur->ne[0];
    const int C = subsampling_channels;
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, W3 * C, H3);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, w.out_w, cur), w.out_b);

    if (out_T_enc)
        *out_T_enc = H3;
    return cur;
}

// ---------------------------------------------------------------------------
// One Conformer encoder block's weights. Bias tensors may be nullptr — when
// they are, the corresponding ggml_add is skipped. This accommodates the
// three FastConformer flavours we ship:
//
//   parakeet    — no biases on Q/K/V/out, ff linears, conv pw1/pw2
//   canary_ctc  — same as parakeet
//   canary      — biases on everything
//
// conv_dw_b is always present (parakeet/canary_ctc populate it synthetically
// via BN folding; canary has the PyTorch bias natively).
// ---------------------------------------------------------------------------
struct BlockWeights {
    // ---- FFN1 (macaron) ----
    ggml_tensor *norm_ff1_w = nullptr, *norm_ff1_b = nullptr;
    ggml_tensor *ff1_l1_w = nullptr, *ff1_l1_b = nullptr;
    ggml_tensor *ff1_l2_w = nullptr, *ff1_l2_b = nullptr;

    // ---- Self-attention (rel-pos with untied u/v biases) ----
    ggml_tensor *norm_attn_w = nullptr, *norm_attn_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    // Fused [Wq;Wk;Wv] concat (set by fuse_qkv at load, CRISPASR_FC_FUSED_QKV).
    // When attn_qkv_w is non-null, build_block does one matmul + view-split.
    ggml_tensor *attn_qkv_w = nullptr, *attn_qkv_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor* attn_pos_w = nullptr; // no bias on rel-pos projection
    ggml_tensor* pos_bias_u = nullptr;
    ggml_tensor* pos_bias_v = nullptr;

    // ---- Conformer convolution module ----
    ggml_tensor *norm_conv_w = nullptr, *norm_conv_b = nullptr;
    ggml_tensor *conv_pw1_w = nullptr, *conv_pw1_b = nullptr; // (2d, d)
    ggml_tensor *conv_dw_w = nullptr, *conv_dw_b = nullptr;   // (d, 1, K)
    ggml_tensor* conv_dw_w_f32 = nullptr;                     // pre-cast F32 copy (set by BN fold)
    ggml_tensor *conv_pw2_w = nullptr, *conv_pw2_b = nullptr; // (d, d)
    // Post-dw-conv LayerNorm affine (NeMo conv_norm_type=layer_norm, e.g.
    // stt_kk_ru hybrid). nullptr for the common batch_norm models, whose
    // BN is folded into conv_dw_w/b at load instead.
    ggml_tensor *conv_ln_w = nullptr, *conv_ln_b = nullptr;

    // ---- FFN2 (macaron) ----
    ggml_tensor *norm_ff2_w = nullptr, *norm_ff2_b = nullptr;
    ggml_tensor *ff2_l1_w = nullptr, *ff2_l1_b = nullptr;
    ggml_tensor *ff2_l2_w = nullptr, *ff2_l2_b = nullptr;

    // ---- Block final LN ----
    ggml_tensor *norm_out_w = nullptr, *norm_out_b = nullptr;
};

// ---------------------------------------------------------------------------
// Load-time Q8_0 repack of the conv pointwise weights (issue #81).
//
// The GGUF stores conv.pw1/pw2 as 3D conv tensors (1, d, 2d)/(1, d, d), so
// crispasr-quantize's 2D-only rule skips them and they ship as F16 even in
// Q8_0/Q4_K models. The ggml CPU F16 mul_mat has no repack fast path and
// measures ~6x slower per FLOP than Q8_0 (M1 per-node profile: the two pw
// matmuls were 35% of encoder time in a q8_0 parakeet-ctc). Repacking them
// to 2D Q8_0 at load moves them onto the optimized int8 kernels.
//
// Gate: CRISPASR_FC_PW_Q8 — "0" forces off, "1" forces on. Unset: enabled
// only when the model is already quantized (pw quantization noise is then
// in-family); pure F16/F32 models keep their exact weights.
// ---------------------------------------------------------------------------
static inline int fc_pw_q8_mode() { // -1 = auto, 0 = off, 1 = on
    static int v = -2;
    if (v == -2) {
        const char* e = std::getenv("CRISPASR_FC_PW_Q8");
        v = (!e || !*e) ? -1 : (*e != '0' ? 1 : 0);
    }
    return v;
}

struct PwRepackBuf {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    void free() {
        if (buf)
            ggml_backend_buffer_free(buf);
        if (ctx)
            ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
    }
};

// Repack each layer's F16 conv_pw1_w / conv_pw2_w into fresh 2D Q8_0 tensors
// (allocated in `out`) and repoint the BlockWeights fields. `model_quantized`
// should be true when the surrounding model weights are quantized (used by
// the auto gate). Returns the number of tensors repacked.
static inline int repack_conv_pw_q8(std::vector<BlockWeights*>& layers, ggml_backend_t backend, bool model_quantized,
                                    PwRepackBuf& out, const char* tag) {
    const int mode = fc_pw_q8_mode();
    if (mode == 0 || (mode == -1 && !model_quantized))
        return 0;

    auto eligible = [](ggml_tensor* t) {
        if (!t || t->type != GGML_TYPE_F16 || !ggml_is_contiguous(t))
            return false;
        const int64_t n_per_row = t->ne[0] > 1 ? t->ne[0] : t->ne[1]; // (d,2d) or 3D (1,d,2d)
        return n_per_row % ggml_blck_size(GGML_TYPE_Q8_0) == 0;
    };

    size_t n_tensors = 0;
    for (auto* e : layers)
        n_tensors += (eligible(e->conv_pw1_w) ? 1 : 0) + (eligible(e->conv_pw2_w) ? 1 : 0);
    if (n_tensors == 0)
        return 0;

    ggml_init_params ip = {n_tensors * ggml_tensor_overhead(), nullptr, true};
    out.ctx = ggml_init(ip);
    if (!out.ctx)
        return 0;

    // Pass 1: create the Q8_0 tensors (2D — collapse the leading unit dim).
    std::vector<std::pair<ggml_tensor**, ggml_tensor*>> jobs; // (slot, q8 tensor)
    for (auto* e : layers) {
        for (ggml_tensor** slot : {&e->conv_pw1_w, &e->conv_pw2_w}) {
            ggml_tensor* src = *slot;
            if (!eligible(src))
                continue;
            const int64_t n_per_row = src->ne[0] > 1 ? src->ne[0] : src->ne[1];
            const int64_t n_rows = ggml_nelements(src) / n_per_row;
            ggml_tensor* q8 = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_per_row, n_rows);
            jobs.push_back({slot, q8});
        }
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        out.free();
        return 0;
    }

    // Pass 2: F16 → F32 → Q8_0, upload, repoint.
    std::vector<ggml_fp16_t> h16;
    std::vector<float> h32;
    std::vector<uint8_t> hq;
    for (auto& j : jobs) {
        ggml_tensor* src = *j.first;
        const int64_t n = ggml_nelements(src);
        const int64_t n_per_row = j.second->ne[0];
        const int64_t n_rows = j.second->ne[1];
        h16.resize(n);
        h32.resize(n);
        ggml_backend_tensor_get(src, h16.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(h16.data(), h32.data(), n);
        hq.resize(ggml_row_size(GGML_TYPE_Q8_0, n_per_row) * n_rows);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, h32.data(), hq.data(), 0, n_rows, n_per_row, nullptr);
        ggml_backend_tensor_set(j.second, hq.data(), 0, hq.size());
        *j.first = j.second;
    }

    fprintf(stderr, "%s: repacked %zu F16 conv pw tensors to Q8_0 (CRISPASR_FC_PW_Q8)\n", tag, jobs.size());
    return (int)jobs.size();
}

// ---------------------------------------------------------------------------
// Load-time Q/K/V weight fusion (issue #81). Concatenates each layer's
// attn_q_w / attn_k_w / attn_v_w (same shape, same type) into one
// (d_in, 3*d_out) tensor so build_block issues a single matmul + view-split
// instead of three matmuls over the same input. Output rows are the same
// independent dot products, so the result is bit-identical to the split path.
//
// Gate: CRISPASR_FC_FUSED_QKV — "0" off, unset/other = on.
// ---------------------------------------------------------------------------
static inline bool fc_fused_qkv_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_FC_FUSED_QKV");
        v = (e && *e == '0') ? 0 : 1;
    }
    return v != 0;
}

static inline int fuse_qkv(std::vector<BlockWeights*>& layers, ggml_backend_t backend, PwRepackBuf& out,
                           const char* tag) {
    if (!fc_fused_qkv_enabled())
        return 0;

    auto eligible = [](const BlockWeights* e) {
        ggml_tensor *q = e->attn_q_w, *k = e->attn_k_w, *v = e->attn_v_w;
        if (!q || !k || !v)
            return false;
        if (q->type != k->type || q->type != v->type)
            return false;
        if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v))
            return false;
        if (q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0] || q->ne[1] != k->ne[1] || q->ne[1] != v->ne[1] ||
            ggml_n_dims(q) != 2 || ggml_n_dims(k) != 2 || ggml_n_dims(v) != 2)
            return false;
        // Biases: all absent or all present (F32 1-D).
        const int nb = (e->attn_q_b != nullptr) + (e->attn_k_b != nullptr) + (e->attn_v_b != nullptr);
        if (nb != 0 && nb != 3)
            return false;
        if (nb == 3 && (e->attn_q_b->type != GGML_TYPE_F32 || e->attn_k_b->type != GGML_TYPE_F32 ||
                        e->attn_v_b->type != GGML_TYPE_F32))
            return false;
        return true;
    };

    size_t n_tensors = 0;
    for (auto* e : layers)
        if (eligible(e))
            n_tensors += e->attn_q_b ? 2 : 1;
    if (n_tensors == 0)
        return 0;

    ggml_init_params ip = {n_tensors * ggml_tensor_overhead(), nullptr, true};
    out.ctx = ggml_init(ip);
    if (!out.ctx)
        return 0;

    struct Job {
        BlockWeights* e;
        ggml_tensor *w = nullptr, *b = nullptr;
    };
    std::vector<Job> jobs;
    for (auto* e : layers) {
        if (!eligible(e))
            continue;
        Job j;
        j.e = e;
        j.w = ggml_new_tensor_2d(out.ctx, e->attn_q_w->type, e->attn_q_w->ne[0], 3 * e->attn_q_w->ne[1]);
        if (e->attn_q_b)
            j.b = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, 3 * e->attn_q_b->ne[0]);
        // cppcheck-suppress uninitvar
        jobs.push_back(j);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!buf) {
        out.free();
        return 0;
    }
    out.buf = buf;

    std::vector<uint8_t> h;
    int n_fused = 0;
    for (auto& j : jobs) {
        size_t off = 0;
        for (ggml_tensor* src : {j.e->attn_q_w, j.e->attn_k_w, j.e->attn_v_w}) {
            const size_t nb = ggml_nbytes(src);
            h.resize(nb);
            ggml_backend_tensor_get(src, h.data(), 0, nb);
            ggml_backend_tensor_set(j.w, h.data(), off, nb);
            off += nb;
        }
        j.e->attn_qkv_w = j.w;
        if (j.b) {
            off = 0;
            for (ggml_tensor* src : {j.e->attn_q_b, j.e->attn_k_b, j.e->attn_v_b}) {
                const size_t nb = ggml_nbytes(src);
                h.resize(nb);
                ggml_backend_tensor_get(src, h.data(), 0, nb);
                ggml_backend_tensor_set(j.b, h.data(), off, nb);
                off += nb;
            }
            j.e->attn_qkv_b = j.b;
        }
        n_fused++;
    }

    fprintf(stderr, "%s: fused Q/K/V projections for %d layers (CRISPASR_FC_FUSED_QKV)\n", tag, n_fused);
    return n_fused;
}

struct BlockParams {
    int d; // d_model
    int n_heads;
    int head_dim; // d / n_heads
    int K;        // conv_kernel (usually 9)
    float ln_eps; // LayerNorm epsilon

    // Local attention window (rel_pos_local_attn). Negative = full attention.
    int att_context_left = -1;  // max positions to the left each query sees
    int att_context_right = -1; // max positions to the right
    int global_tokens = 0;      // first N positions are global (visible to all)

    // Use the manual QK^T + soft_max_ext + V attention instead of
    // flash_attn_ext (set from fc_gpu_manual_attn — CUDA rejects the
    // per-head rel-pos mask and would bounce every flash node to CPU).
    bool manual_attn = false;
};

// TRUE windowed (block sliding-chunks) attention for rel_pos_local_attn models:
// scores and rel-pos bias are computed only within a local band, giving
// O(T·window) memory instead of the O(T²) masked-full attention. Bit-exact vs
// masked-full (tools/dev/winattn_parity.cpp) and ~3× faster on Metal.
// DEFAULT ON when --att-context is set (only engages via build_block when a band
// mask is supplied and T >= 2*BS). Set CRISPASR_FC_WINDOWED_ATTN=0 to force the
// legacy masked-full local path (T×T mask over full attention) for A/B.
static inline bool fc_windowed_attn() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_FC_WINDOWED_ATTN");
        v = (e && *e == '0') ? 0 : 1; // default ON; only an explicit "0" disables
    }
    return v != 0;
}

// Block size for windowed attention. Must be >= max(att_left, att_right) so the
// 3-block band [b-1, b, b+1] covers every query's window. Shared between the
// encoder (mask builder) and build_block so the band mask dims agree. An env
// override (CRISPASR_FC_WINDOW_BLOCK) can trade graph size vs. band width.
static inline int fc_window_block_size(int att_left, int att_right) {
    int bs = att_left > att_right ? att_left : att_right;
    if (bs < 1)
        bs = 1;
    const char* e = std::getenv("CRISPASR_FC_WINDOW_BLOCK");
    if (e && *e) {
        int ov = std::atoi(e);
        if (ov >= bs)
            bs = ov; // only allow >= the minimum that still covers the window
    }
    return bs;
}

// Whether the windowed path can run for this (T, window). Needs T >= 2*BS so the
// rel-pos R slice [T-2BS .. T+2BS-2] stays in range; short clips fall back to
// masked-full (cheap anyway).
static inline bool fc_window_attn_applicable(int T, int att_left, int att_right) {
    if (att_left < 0 || att_right < 0)
        return false;
    const int bs = fc_window_block_size(att_left, att_right);
    return T >= 2 * bs && T > att_left + att_right + 1;
}

// True windowed (block sliding-chunks) rel-pos attention. Inputs are the
// post-projection tensors as they exist in build_block right after the
// permutes: Q_u, Q_v, K_ are (head_dim, T, n_heads); V3 is (head_dim, n_heads,
// T); R is (head_dim, 2T-1, n_heads). `band_mask` is a host-filled (3*BS, BS,
// NB) F32 additive mask (0 visible / -inf masked, natural key order). Returns
// attn_out (d, T). Validated bit-exact vs masked-full in tools/dev/winattn_parity.cpp.
static inline ggml_tensor* build_windowed_attn(ggml_context* ctx0, ggml_tensor* Q_u, ggml_tensor* Q_v, ggml_tensor* K_,
                                               ggml_tensor* V3, ggml_tensor* R, int T, const BlockParams& p,
                                               ggml_tensor* band_mask) {
    const int HD = p.head_dim;
    const int NH = p.n_heads;
    const int d = p.d;
    const int BS = fc_window_block_size(p.att_context_left, p.att_context_right);
    const int NB = (T + BS - 1) / BS;
    const int Tp = NB * BS;
    const int RB = 4 * BS - 1;
    const float scale = 1.0f / sqrtf((float)HD);

    // Head-shaped, contiguous. V3 (HD,NH,T) -> (HD,T,NH).
    ggml_tensor* Qu = ggml_cont(ctx0, Q_u);                                // (HD,T,NH)
    ggml_tensor* Qv = ggml_cont(ctx0, Q_v);                                // (HD,T,NH)
    ggml_tensor* Kc = ggml_cont(ctx0, K_);                                 // (HD,T,NH)
    ggml_tensor* Vc = ggml_cont(ctx0, ggml_permute(ctx0, V3, 0, 2, 1, 3)); // (HD,T,NH)

    // Zero-pad queries to Tp (tail) and keys/values to Tp+2*BS (band halo + tail).
    auto pad_q = [&](ggml_tensor* xin) {
        return (Tp == T) ? xin : ggml_pad_ext(ctx0, xin, 0, 0, 0, Tp - T, 0, 0, 0, 0);
    };
    ggml_tensor* Qu_p = ggml_cont(ctx0, pad_q(Qu)); // (HD,Tp,NH)
    ggml_tensor* Qv_p = ggml_cont(ctx0, pad_q(Qv));
    // left BS, right BS + (Tp - T)  -> length T + 2*BS + (Tp-T) = Tp + 2*BS
    auto pad_kv = [&](ggml_tensor* xin) { return ggml_pad_ext(ctx0, xin, 0, 0, BS, BS + (Tp - T), 0, 0, 0, 0); };
    ggml_tensor* Kp = ggml_cont(ctx0, pad_kv(Kc)); // (HD,Tp+2*BS,NH)
    ggml_tensor* Vp = ggml_cont(ctx0, pad_kv(Vc));

    // 3-block band via non-overlapping block reshape + 3 stride-1 slices + concat
    // (ggml forbids overlapping views). block bo keys = [orig bo-1, bo, bo+1].
    auto band3 = [&](ggml_tensor* Xp) {
        ggml_tensor* Xb = ggml_reshape_4d(ctx0, Xp, HD, BS, NB + 2, NH);
        auto slc = [&](int i) {
            return ggml_cont(
                ctx0, ggml_view_4d(ctx0, Xb, HD, BS, NB, NH, Xb->nb[1], Xb->nb[2], Xb->nb[3], (size_t)i * Xb->nb[2]));
        };
        return ggml_concat(ctx0, ggml_concat(ctx0, slc(0), slc(1), 1), slc(2), 1); // (HD,3BS,NB,NH)
    };
    ggml_tensor* K_band = band3(Kp);
    ggml_tensor* V_band = band3(Vp);

    // Query blocks (HD,BS,NB,NH).
    ggml_tensor* Qu_blk = ggml_cont(
        ctx0, ggml_view_4d(ctx0, Qu_p, HD, BS, NB, NH, Qu_p->nb[1], (size_t)BS * Qu_p->nb[1], Qu_p->nb[2], 0));
    ggml_tensor* Qv_blk = ggml_cont(
        ctx0, ggml_view_4d(ctx0, Qv_p, HD, BS, NB, NH, Qv_p->nb[1], (size_t)BS * Qv_p->nb[1], Qv_p->nb[2], 0));

    // Banded AC term: scores[j,i,b,h] = <K_band key j, Q_u query i>.
    ggml_tensor* sc = ggml_mul_mat(ctx0, K_band, Qu_blk); // (3BS,BS,NB,NH)

    // Banded rel-pos bias BD. R_sl = R rows [T-2BS .. T-2BS+RB-1] reshaped to
    // (HD,RB,1,NH) so mul_mat broadcasts R over blocks and aligns heads in ne3.
    ggml_tensor* R_sl =
        ggml_cont(ctx0, ggml_view_3d(ctx0, R, HD, RB, NH, R->nb[1], R->nb[2], (size_t)(T - 2 * BS) * R->nb[1]));
    R_sl = ggml_reshape_4d(ctx0, R_sl, HD, RB, 1, NH);
    ggml_tensor* BDraw_blk = ggml_mul_mat(ctx0, R_sl, Qv_blk); // (RB,BS,NB,NH)
    // In-block rel-shift: BD_blk[j,i] = BDraw_blk[(BS-1)+j-i, i] (natural key order).
    ggml_tensor* BD_blk =
        ggml_cont(ctx0, ggml_view_4d(ctx0, BDraw_blk, 3 * BS, BS, NB, NH, BDraw_blk->nb[1] - BDraw_blk->nb[0],
                                     BDraw_blk->nb[2], BDraw_blk->nb[3], (size_t)(BS - 1) * BDraw_blk->nb[0]));
    sc = ggml_add(ctx0, sc, BD_blk);

    // Band mask (3BS,BS,NB) broadcast over heads.
    sc = ggml_add(ctx0, sc, ggml_reshape_4d(ctx0, band_mask, 3 * BS, BS, NB, 1));
    sc = ggml_soft_max_ext(ctx0, sc, nullptr, scale, 0.0f);

    // Weighted sum over the band: attn[:,i,b] = sum_j sc[j,i,b] * V_band[:,j,b].
    ggml_tensor* V_band_t = ggml_cont(ctx0, ggml_permute(ctx0, V_band, 1, 0, 2, 3)); // (3BS,HD,NB,NH)
    ggml_tensor* a_blk = ggml_mul_mat(ctx0, V_band_t, sc);                           // (HD,BS,NB,NH)
    ggml_tensor* a_full = ggml_cont(ctx0, ggml_reshape_3d(ctx0, ggml_cont(ctx0, a_blk), HD, Tp, NH));
    if (Tp != T)
        a_full = ggml_cont(ctx0, ggml_view_3d(ctx0, a_full, HD, T, NH, a_full->nb[1], a_full->nb[2], 0));
    // (HD,T,NH) -> (d,T)
    return ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, a_full, 0, 2, 1, 3)), d, T);
}

// Env gate: CRISPASR_FC_TILED_ATTN=1 enables query-TILED full attention for
// rel_pos (full-attention) models — EXACT (bit-identical) output, but the rel-pos
// bias BD (and, on the manual path, the QK^T scores) is computed one query-block
// at a time so peak memory is O(T·block) instead of O(T²). Default OFF; only for
// pure full attention (no local/pad mask). Bit-exact vs monolithic
// (tools/dev/tiledbd_parity.cpp).
//
// WHERE IT HELPS: the MANUAL attention path (CUDA, fc_gpu_manual_attn default-on),
// which materializes O(T²) scores+BD at large single-pass T — the ~2 GiB VRAM in
// issue #257. It does NOT help Metal/flash (flash_attn_ext never materializes the
// scores, so full-attention memory is already ~flat with length there) and is
// slower than flash — so it is a CUDA-side memory lever, gated off by default.
// Validate on CUDA (tools/kaggle/windowed-attn-cuda).
static inline bool fc_tiled_attn() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_FC_TILED_ATTN");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Query-block size for tiled attention. Bigger = fewer blocks / graph nodes but
// larger O(T·block) slab; smaller = less memory but more launches. Env override
// CRISPASR_FC_TILED_BLOCK.
static inline int fc_tiled_block_size() {
    int bs = 512;
    if (const char* e = std::getenv("CRISPASR_FC_TILED_BLOCK")) {
        int o = std::atoi(e);
        if (o >= 1)
            bs = o;
    }
    return bs;
}

// Tiled only helps when there is more than one block; short clips run the cheap
// monolithic path.
static inline bool fc_tiled_attn_applicable(int T) {
    return T > fc_tiled_block_size();
}

// Query-TILED full rel-pos attention. Same inputs/shapes as build_windowed_attn.
// Processes queries in blocks of BS against ALL keys, computing only a (T×BS)
// rel-pos bias slab per block (the O(T²) hog becomes O(T·BS) peak). Bit-exact to
// monolithic full attention. Manual QK^T path (works on every backend).
static inline ggml_tensor* build_tiled_attn(ggml_context* ctx0, ggml_tensor* Q_u, ggml_tensor* Q_v, ggml_tensor* K_,
                                            ggml_tensor* V3, ggml_tensor* R, int T, const BlockParams& p) {
    const int HD = p.head_dim;
    const int NH = p.n_heads;
    const int d = p.d;
    const int BS = fc_tiled_block_size();
    const int NB = (T + BS - 1) / BS;
    const int Tp = NB * BS;
    const float scale = 1.0f / sqrtf((float)HD);

    ggml_tensor* Qu = ggml_cont(ctx0, Q_u);                                // (HD,T,NH)
    ggml_tensor* Qv = ggml_cont(ctx0, Q_v);                                // (HD,T,NH)
    ggml_tensor* Kh = ggml_cont(ctx0, K_);                                 // (HD,T,NH)
    ggml_tensor* Vh = ggml_cont(ctx0, ggml_permute(ctx0, V3, 0, 2, 1, 3)); // (HD,T,NH)
    ggml_tensor* Vt = ggml_cont(ctx0, ggml_permute(ctx0, Vh, 1, 0, 2, 3)); // (T,HD,NH) shared across blocks
    // Pad queries to Tp (padded queries are sliced off at the end).
    ggml_tensor* Qup = (Tp == T) ? Qu : ggml_cont(ctx0, ggml_pad_ext(ctx0, Qu, 0, 0, 0, Tp - T, 0, 0, 0, 0));
    ggml_tensor* Qvp = (Tp == T) ? Qv : ggml_cont(ctx0, ggml_pad_ext(ctx0, Qv, 0, 0, 0, Tp - T, 0, 0, 0, 0));

    ggml_tensor* out = nullptr; // (HD, *, NH), concatenated per block
    for (int b = 0; b < NB; b++) {
        ggml_tensor* Qu_b =
            ggml_cont(ctx0, ggml_view_3d(ctx0, Qup, HD, BS, NH, Qup->nb[1], Qup->nb[2], (size_t)b * BS * Qup->nb[1]));
        ggml_tensor* Qv_b =
            ggml_cont(ctx0, ggml_view_3d(ctx0, Qvp, HD, BS, NH, Qvp->nb[1], Qvp->nb[2], (size_t)b * BS * Qvp->nb[1]));
        ggml_tensor* BDraw_b = ggml_mul_mat(ctx0, R, Qv_b); // (2T-1,BS,NH)
        // BD_b[k,i] = BDraw_b[k - i + (T-1) - b*BS, i]: rel_shift with offset (T-1-b*BS).
        // Offset >= 0 for BS>=2 (b*BS <= Tp-BS <= T-1); rows stay within [0,2T-2].
        const int off = (T - 1) - b * BS;
        ggml_tensor* BD_b = ggml_view_3d(ctx0, BDraw_b, T, BS, NH, BDraw_b->nb[1] - BDraw_b->nb[0], BDraw_b->nb[2],
                                         (size_t)off * BDraw_b->nb[0]);
        ggml_tensor* sc = ggml_mul_mat(ctx0, Kh, Qu_b); // (T,BS,NH)
        sc = ggml_add(ctx0, sc, ggml_cont(ctx0, BD_b));
        sc = ggml_soft_max_ext(ctx0, sc, nullptr, scale, 0.0f);
        ggml_tensor* ob = ggml_mul_mat(ctx0, Vt, sc); // (HD,BS,NH)
        out = out ? ggml_concat(ctx0, out, ob, 1) : ob;
    }
    if (Tp != T)
        out = ggml_view_3d(ctx0, out, HD, T, NH, out->nb[1], out->nb[2], 0);
    return ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, ggml_cont(ctx0, out), 0, 2, 1, 3)), d, T);
}

// Build one Conformer block. `cur` must be (d, T). `pos_enc` is the shared
// sinusoidal rel-pos table (d, 2T-1). `local_attn_mask` is an optional (T, T)
// F32 tensor with 0.0 for visible positions and -inf for masked positions
// (used by rel_pos_local_attn models, and by the bucketed-padding path to
// mask pad keys). Pass nullptr for full attention.
// `time_mask` is an optional (1, T) F32 multiplicative mask (1.0 valid /
// 0.0 pad) applied before the depthwise conv and at the block output —
// with pad keys masked in attention (-inf; a finite constant gets overrun
// once pad garbage grows), these cover every inter-column path, so valid
// columns match an unpadded graph exactly up to GEMM micro-kernel ULP
// reassociation (see the CRISPASR_FC_BUCKET note in canary_ctc.cpp).
// Returns the post-block (d, T) output.
static inline ggml_tensor* build_block(ggml_context* ctx0, ggml_tensor* cur, ggml_tensor* pos_enc, int T,
                                       const BlockWeights& e, const BlockParams& p,
                                       ggml_tensor* local_attn_mask = nullptr, ggml_tensor* time_mask = nullptr,
                                       ggml_tensor* window_band_mask = nullptr) {
    const int d = p.d;
    const int n_heads = p.n_heads;
    const int head_dim = p.head_dim;
    const int K = p.K;
    const float eps = p.ln_eps;

    // Tiny helper: mul_mat + optional bias add.
    auto mm_bias = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };

    ggml_tensor* inpL = cur;

    // ---- FFN1 (macaron half) ----
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, e.norm_ff1_w, e.norm_ff1_b, eps);
    x = mm_bias(e.ff1_l1_w, x, e.ff1_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff1_l2_w, x, e.ff1_l2_b);
    cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));

    ggml_tensor* inpAttn = cur;

    // ---- Self-Attention (rel_pos with untied biases) ----
    x = ggml_norm_affine(ctx0, cur, e.norm_attn_w, e.norm_attn_b, eps);

    // Q/K/V projections — fused single matmul + view-split when the load-time
    // concat is available (bit-identical: each output row is the same dot
    // product either way), else three separate matmuls.
    ggml_tensor* Q;  // (d, T)
    ggml_tensor* K3; // (head_dim, n_heads, T)
    ggml_tensor* V3; // (head_dim, n_heads, T)
    if (e.attn_qkv_w) {
        ggml_tensor* qkv = ggml_mul_mat(ctx0, e.attn_qkv_w, x); // (3d, T)
        if (e.attn_qkv_b)
            qkv = ggml_add(ctx0, qkv, e.attn_qkv_b);
        Q = ggml_view_2d(ctx0, qkv, d, T, qkv->nb[1], 0);
        K3 = ggml_view_3d(ctx0, qkv, head_dim, n_heads, T, (size_t)head_dim * sizeof(float), qkv->nb[1],
                          (size_t)d * sizeof(float));
        V3 = ggml_view_3d(ctx0, qkv, head_dim, n_heads, T, (size_t)head_dim * sizeof(float), qkv->nb[1],
                          (size_t)2 * d * sizeof(float));
    } else {
        Q = mm_bias(e.attn_q_w, x, e.attn_q_b);
        K3 = ggml_reshape_3d(ctx0, mm_bias(e.attn_k_w, x, e.attn_k_b), head_dim, n_heads, T);
        V3 = ggml_reshape_3d(ctx0, mm_bias(e.attn_v_w, x, e.attn_v_b), head_dim, n_heads, T);
    }
    ggml_tensor* R = ggml_mul_mat(ctx0, e.attn_pos_w, pos_enc); // no bias

    ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_u, d));
    ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_v, d));

    Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);
    Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);
    ggml_tensor* K_ = ggml_permute(ctx0, K3, 0, 2, 1, 3);
    R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

    const float scale = 1.0f / sqrtf((float)head_dim);

    ggml_tensor* attn_out;
    // ---- TRUE windowed attention (block sliding-chunks, O(T·window) memory) ----
    if (fc_windowed_attn() && window_band_mask &&
        fc_window_attn_applicable(T, p.att_context_left, p.att_context_right)) {
        static bool logged = false;
        if (!logged && std::getenv("CRISPASR_FC_MEM_DEBUG")) {
            logged = true;
            fprintf(stderr, "[fc] windowed attn ENGAGED: T=%d BS=%d left=%d right=%d\n", T,
                    fc_window_block_size(p.att_context_left, p.att_context_right), p.att_context_left,
                    p.att_context_right);
        }
        attn_out = build_windowed_attn(ctx0, Q_u, Q_v, K_, V3, R, T, p, window_band_mask);
    } else if (fc_tiled_attn() && !local_attn_mask && fc_tiled_attn_applicable(T)) {
        // ---- Query-TILED full attention (exact, O(T·block) peak memory) ----
        static bool logged = false;
        if (!logged && std::getenv("CRISPASR_FC_MEM_DEBUG")) {
            logged = true;
            fprintf(stderr, "[fc] tiled attn ENGAGED: T=%d BS=%d\n", T, fc_tiled_block_size());
        }
        attn_out = build_tiled_attn(ctx0, Q_u, Q_v, K_, V3, R, T, p);
    } else {
        // Compute the relative position bias BD = rel_shift(Q_v × R^T).
        // This is query-dependent so it can't be precomputed, but it CAN
        // be passed as the additive mask to ggml_flash_attn_ext, which
        // fuses AC (= Q_u × K^T) + BD + softmax + ×V into one kernel.
        ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
        ggml_tensor* BD = rel_shift(ctx0, BD_raw);

        // flash_attn_ext computes: softmax(Q_u × K^T * scale + mask) × V
        // We need:                 softmax((Q_u × K^T + BD) * scale)  × V
        // So pass mask = BD * scale to get equivalent semantics.
        // BD is a strided view from rel_shift — make contiguous before scale/cast.
        ggml_tensor* BD_c = ggml_cont(ctx0, BD);
        ggml_tensor* BD_scaled = ggml_scale(ctx0, BD_c, scale);

        // Local attention window mask: add -inf for positions outside the window.
        // The mask tensor is created externally and passed via `local_attn_mask`.
        if (local_attn_mask) {
            // Broadcast (T, T) mask across all heads in BD_scaled (T, T, n_heads).
            BD_scaled = ggml_add(ctx0, BD_scaled, local_attn_mask);
        }

        if (fc_no_flash() || p.manual_attn) {
            // Manual attention: QK^T + BD, softmax, ×V — no flash_attn_ext.
            // Q_u, K_ are (head_dim, T, n_heads) after permute.
            // mul_mat(K, Q) computes Q^T × K^T^T = Q^T × K → (T, T, n_heads).
            ggml_tensor* Q_u_c = ggml_cont(ctx0, Q_u);            // (head_dim, T, n_heads)
            ggml_tensor* K_c = ggml_cont(ctx0, K_);               // (head_dim, T, n_heads)
            ggml_tensor* scores = ggml_mul_mat(ctx0, K_c, Q_u_c); // (T, T, n_heads)
            ggml_tensor* BD_c2 = ggml_cont(ctx0, BD);             // (T, T, n_heads)
            scores = ggml_add(ctx0, scores, BD_c2);
            if (local_attn_mask)
                scores = ggml_add(ctx0, scores, local_attn_mask);
            scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);
            // V: (head_dim, n_heads, T) → permute(0,2,1,3) → (head_dim, T, n_heads)
            // Then transpose via permute(1,0,2,3) → (T, head_dim, n_heads) for mul_mat
            ggml_tensor* V_3d = ggml_cont(ctx0, ggml_permute(ctx0, V3, 0, 2, 1, 3));
            // V_3d: (head_dim, T, n_heads) — need (T, head_dim, n_heads) for mul_mat
            ggml_tensor* V_t = ggml_cont(ctx0, ggml_permute(ctx0, V_3d, 1, 0, 2, 3)); // (T, head_dim, n_heads)
            attn_out = ggml_mul_mat(ctx0, V_t, scores);                               // (head_dim, T, n_heads)
            attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);
        } else {
            // flash_attn_ext mask must be F16
            ggml_tensor* BD_mask = ggml_cast(ctx0, BD_scaled, GGML_TYPE_F16);

            // V needs [head_dim, T, n_heads] layout for flash_attn_ext (same as K).
            // The kernel reads strided views (nb0 == type size) directly — the
            // legacy ggml_cont copies of Q/K/V are restorable via
            // CRISPASR_FC_ATTN_CONT=1 for regression bisection.
            ggml_tensor* Q_f = Q_u;
            ggml_tensor* K_f = K_;
            ggml_tensor* V_f = ggml_permute(ctx0, V3, 0, 2, 1, 3);
            if (fc_attn_cont()) {
                Q_f = ggml_cont(ctx0, Q_f);
                K_f = ggml_cont(ctx0, K_f);
                V_f = ggml_cont(ctx0, V_f);
            }

            attn_out = ggml_flash_attn_ext(ctx0, Q_f, K_f, V_f, BD_mask, scale, 0.0f, 0.0f);
            attn_out = ggml_reshape_2d(ctx0, attn_out, d, T);
        }
    }

    attn_out = mm_bias(e.attn_out_w, attn_out, e.attn_out_b);
    cur = ggml_add(ctx0, inpAttn, attn_out);

    // ---- Conformer convolution module ----
    ggml_tensor* inpConv = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_conv_w, e.norm_conv_b, eps);

    // pw1: (d → 2d), then sigmoid GLU — fused into one op, avoids strided-view
    // CUDA fallback that plagued the manual sigmoid path (see issue #81 PR #05).
    ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
    ggml_tensor* cnv = mm_bias(pw1_w, x, e.conv_pw1_b);
    cnv = ggml_siglu_swapped(ctx0, cnv);
    if (time_mask) // zero pad columns so the dw conv sees them as its own zero border
        cnv = ggml_mul(ctx0, cnv, time_mask);

    // dw conv (kernel K, padding K/2). BN was folded into conv_dw_w/b at load.
    // Use pre-cast F32 weights if available (avoids F16→F32 cast per forward).
    ggml_tensor* dw_w_f32 = e.conv_dw_w_f32 ? e.conv_dw_w_f32 : ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
    ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);
    cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv)); // (d, T) → (T, d)
    cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
    cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (K - 1) / 2, 0, 1, 1);
    cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
    cnv = ggml_reshape_2d(ctx0, cnv, d, T);

    cnv = ggml_add(ctx0, cnv, ggml_reshape_2d(ctx0, e.conv_dw_b, d, 1));
    if (e.conv_ln_w) // conv_norm_type=layer_norm: LN over channels per frame
        cnv = ggml_norm_affine(ctx0, cnv, e.conv_ln_w, e.conv_ln_b, eps);
    cnv = ggml_silu(ctx0, cnv);

    // pw2: (d → d)
    ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
    cnv = mm_bias(pw2_w, cnv, e.conv_pw2_b);
    cur = ggml_add(ctx0, inpConv, cnv);

    // ---- FFN2 (macaron half) ----
    ggml_tensor* inpFF2 = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_ff2_w, e.norm_ff2_b, eps);
    x = mm_bias(e.ff2_l1_w, x, e.ff2_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff2_l2_w, x, e.ff2_l2_b);
    cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, x, 0.5f));

    // ---- Block final LN ----
    cur = ggml_norm_affine(ctx0, cur, e.norm_out_w, e.norm_out_b, eps);

    // Re-zero pad columns at the block boundary (bucketed path): pad garbage
    // otherwise grows across blocks until 0-weight × huge-value artifacts
    // (or a mask-magnitude overrun) leak into valid columns.
    if (time_mask)
        cur = ggml_mul(ctx0, cur, time_mask);

    return cur;
}

} // namespace core_conformer
