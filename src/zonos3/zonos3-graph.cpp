// Mixed GDN/global GGUF inference, Full AttnRes, and continuous batching.
#include "zonos3.h"
#include "zonos3-sampler.h"
#include "zonos3-steer.h"
#include "zonos3-accent.h"
#include "npy.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Graph inputs and outputs of the optional features: accent adapters (prefill and decode) and the decode-step
// alignment probe (readout over each slot's text keys; masked-head recomputation of up to `masked` slots).
struct zonos3_feature_io {
    bool accent = false;
    ggml_tensor* ac_mask = nullptr;         // [n*K, rows] adapter strength on the row's adapter columns
    ggml_tensor* ac_emask = nullptr;        // [n*R, 1, rows] the same for expert factors
    std::map<int, ggml_tensor*> ac_rbias;   // MoE layer -> [E, rows] router balancing-bias shift
    int lmax = 0, masked = 0;               // probe text keys per slot; masked-probe slots (0 = none)
    std::vector<int> attn_layers;           // global layers in probe order
    ggml_tensor* pr_kidx = nullptr;         // I32 [lmax*width, n_kv]: cache rows of each column's text keys
    ggml_tensor* pr_tpos = nullptr;         // [lmax] text indices 0..lmax-1
    std::vector<ggml_tensor*> pr_w;         // per global layer [n_head, 1, width] head weights
    std::vector<ggml_tensor*> pr_lo, pr_hi; // per global layer [1, 1, n_head, width] inclusive text windows
    ggml_tensor* pr_acc = nullptr;          // output [lmax, width] weighted text distribution
    ggml_tensor* pm_rows = nullptr;         // I32 [masked] decode columns of the masked slots
    ggml_tensor* pm_kidx = nullptr;         // I32 [masked*slot_cap, n_kv]: cache rows of those slots' keys
    ggml_tensor* pm_kpos = nullptr;         // [slot_cap] key positions 0..slot_cap-1
    ggml_tensor *pm_npast = nullptr, *pm_ts = nullptr, *pm_tl = nullptr; // [1, 1, 1, masked]
    std::vector<ggml_tensor*> pm_flag;      // per global layer [1, n_head, masked]: 1 = masked head
    std::vector<ggml_tensor*> pm_lo, pm_hi; // per global layer [1, 1, n_head, masked] text windows
};

namespace {

// An accent selection takes effect only with a loaded bank and a nonzero strength.
bool accent_on(const zonos3_model& m, int accent, float strength) {
    return accent >= 0 && strength != 0.0f && m.accents && accent < m.accents->n;
}

struct gctx {
    const zonos3_model* m = nullptr;
    ggml_context* ctx = nullptr;
    int n = 0;
    std::vector<ggml_tensor*> ids;
    ggml_tensor *pos_rope = nullptr, *pos_cache = nullptr;
    bool capture = true, cap_moe = false, decode = false;
    std::vector<std::pair<std::string, ggml_tensor*>> caps;
    std::vector<ggml_tensor*> zero_inputs;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> updates;
    std::vector<ggml_tensor*>*kc = nullptr, *vc = nullptr;
    std::vector<ggml_tensor*>*state = nullptr, *history = nullptr;
    int n_slots = 1, slot_cap = 0, cache_slot = 0, position = 0;
    ggml_tensor *active = nullptr, *mask = nullptr, *mask_causal = nullptr;
    ggml_cgraph* gf = nullptr;
    const zonos3_conditioning* cond = nullptr;
    // Prefill rows of one or more sequences (a CFG pair): segment s owns graph rows [row0, row0 + n) at absolute rows
    // [position, position + n) of cache slot cache_slot. Row-local ops run over all rows; attention and GDN per segment.
    struct segment {
        int row0, n, cache_slot, position;
        const zonos3_conditioning* cond;
    };
    std::vector<segment> segs; // prefill only; empty = the single segment (0, n, cache_slot, position, cond)
    std::vector<segment> segments() const {
        return segs.empty() ? std::vector<segment>{{0, n, cache_slot, position, cond}} : segs;
    }
    std::vector<ggml_tensor*> masks_causal; // per segment [position + n, n]; mask_causal = masks_causal[0]
    // Row replacement (speaker / condition rows): inputs exist only when the chunk contains such a row. Table column 0
    // is zero, then one speaker column per segment with its speaker row in the chunk, then n_cond bucket columns per
    // segment with condition rows in the chunk.
    ggml_tensor *ovr_keep = nullptr, *ovr_sel = nullptr, *ovr_zero = nullptr;
    std::vector<ggml_tensor*> ovr_spks, ovr_bkts; // inputs in segment order
    std::vector<int> ovr_spk_col, ovr_cond_col;   // per segment: first table column, -1 = none
    std::vector<int> steer_sources;               // decode only: bank sources that take a delta input
    std::vector<ggml_tensor*> steer;              // [D, n] input per entry of steer_sources
    std::shared_ptr<zonos3_feature_io> io;        // optional feature inputs (accent adapters, alignment probe)
    ggml_tensor* probe_acc = nullptr;             // running probe readout over the global layers
    std::vector<ggml_tensor*> extra_outputs;      // feature outputs outside the logits' ancestry

    ggml_tensor* out_named(const std::string& name, ggml_tensor* t) {
        // A view marked output does not pin its allocator-owned base. Capture an
        // independent snapshot; generation never takes this diagnostic path.
        t = ggml_dup(ctx, t);
        ggml_set_output(t);
        ggml_set_name(t, name.c_str());
        caps.emplace_back(name, t);
        return t;
    }
    ggml_tensor* cap(const std::string& name, ggml_tensor* t) { return capture ? out_named(name, t) : t; }
    void finish(ggml_tensor* output) {
        ggml_set_output(output);
        ggml_build_forward_expand(gf, output);
        for (const auto& c : caps)
            ggml_build_forward_expand(gf, c.second);
        for (ggml_tensor* t : extra_outputs)
            ggml_build_forward_expand(gf, t);
        // Commit only after every use of the previous state. Persistent state is
        // never an allocator-owned temporary, including CUDA-graph replay.
        for (const auto& u : updates)
            ggml_build_forward_expand(gf, ggml_cpy(ctx, u.first, u.second));
    }
};

// Fill an [n_kv, n_q] F16 causal additive mask: 0 where key kv <= query q, -inf above the
// diagonal. Element (kv, q) lives at index q*n + kv. Positions are the contiguous 0..n-1 used
// by every prefill / non-decode graph, so the same lower-triangular mask serves all of them.
static void set_causal_mask(ggml_tensor* t, int queries, int position = 0) {
    const int keys = (int)t->ne[0];
    std::vector<ggml_fp16_t> mk((size_t)keys * queries);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < queries; ++q)
        for (int kv = 0; kv < keys; ++kv)
            mk[(size_t)q * keys + kv] = kv <= position + q ? zero : ninf;
    ggml_backend_tensor_set(t, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
}

// KV-cache row width: head_dim zero-padded to the smallest flash-attention kernel width (64).
static int zonos3_kv_head_dim(const zonos3_hparams& hp) {
    return std::max((int)hp.head_dim, 64);
}

// The trained reference disables TF32; stored F16 weights do not authorize
// rounding F32 activations or accumulation in the backbone.
static ggml_tensor* matmul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* out = ggml_mul_mat(ctx, a, b);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    return out;
}
static ggml_tensor* matmul_id(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* ids) {
    ggml_tensor* out = ggml_mul_mat_id(ctx, a, b, ids);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    return out;
}

// RMSNorm with learnable weight (w may be null for the weightless emb_norm).
static ggml_tensor* rms_w(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    ggml_tensor* t = ggml_rms_norm(ctx, x, eps);
    return w ? ggml_mul(ctx, t, w) : t;
}

// Linear projection plus the accent-adapter shift of its module (zonos3-accent.h): y += U_all @ ((D_all @ x) * mask).
// `base` is the module's GGUF base tensor name; x is [in, rows] with rows matching the mask.
static ggml_tensor* lin(gctx& g, ggml_tensor* w, ggml_tensor* x, const std::string& base) {
    ggml_tensor* y = matmul(g.ctx, w, x);
    if (!g.io || !g.io->accent || !g.io->ac_mask)
        return y;
    const auto it = g.m->accents->linear.find(base);
    if (it == g.m->accents->linear.end())
        return y;
    ggml_tensor* t = ggml_mul(g.ctx, matmul(g.ctx, it->second.down, x), g.io->ac_mask);
    return ggml_add(g.ctx, y, matmul(g.ctx, it->second.up, t));
}

// Decode-step alignment probe at global layer `il` (sgl align_probe._readout / _masked; zonos3.tts.probe.readout).
// q4 [hd_kv, 1, nh, width] are the decode queries (RoPE, QK-norm and temperature applied, zero-padded like the cache);
// kc/vc the layer's caches, already holding this step's keys; out [hd, nh, width] the attention output. Adds the
// layer's weighted text distribution to g.probe_acc and returns `out` with the masked heads of the masked slots
// recomputed. Explicit F32 arithmetic on every KV profile. Windows become additive masks of 0 / -1e30, which give the
// same softmax as -inf (each row keeps at least one finite key).
static ggml_tensor* build_probe(gctx& g, ggml_tensor* q4, ggml_tensor* kc, ggml_tensor* vc, ggml_tensor* out, int il) {
    auto& io = *g.io;
    const auto& hp = g.m->hp;
    ggml_context* ctx = g.ctx;
    const int hd = (int)hp.head_dim, nh = (int)hp.n_head, nkv = (int)hp.n_head_kv, hd_kv = (int)q4->ne[0];
    const int W = g.n_slots, lmax = io.lmax;
    const auto at = std::find(io.attn_layers.begin(), io.attn_layers.end(), il);
    if (at == io.attn_layers.end())
        return out;
    const size_t s = (size_t)(at - io.attn_layers.begin());
    const float scale = 1.0f / sqrtf((float)hd);
    // Integral x: ge0 = (x >= 0), le0 = (x <= 0), as 0/1 floats.
    auto ge0 = [&](ggml_tensor* x) { return ggml_step(ctx, ggml_scale_bias(ctx, x, 1.0f, 0.5f)); };
    auto le0 = [&](ggml_tensor* x) { return ggml_step(ctx, ggml_scale_bias(ctx, x, -1.0f, 0.5f)); };
    auto additive = [&](ggml_tensor* allowed) { return ggml_scale_bias(ctx, allowed, 1e30f, -1e30f); };

    // Readout: per head softmax over the text keys inside its window, weighted sum over heads.
    ggml_tensor* keys = ggml_get_rows(ctx, kc, io.pr_kidx); // [hd_kv, lmax*W, nkv]
    keys = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, keys, hd_kv, lmax, W, nkv), 0, 1, 3, 2));
    ggml_tensor* scores = matmul(ctx, keys, q4); // [lmax, 1, nh, W]
    ggml_tensor* pos = ggml_repeat(ctx, io.pr_tpos, scores);
    ggml_tensor* allowed = ggml_mul(ctx, ge0(ggml_sub(ctx, pos, io.pr_lo[s])), le0(ggml_sub(ctx, pos, io.pr_hi[s])));
    ggml_tensor* p = ggml_soft_max_ext(ctx, scores, additive(allowed), scale, 0.0f);
    p = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, p, lmax, nh, W), 1, 0, 2, 3)); // [nh, lmax, W]
    ggml_tensor* acc = ggml_reshape_2d(ctx, matmul(ctx, p, io.pr_w[s]), lmax, W);
    g.probe_acc = g.probe_acc ? ggml_add(ctx, g.probe_acc, acc) : acc;
    if (!io.masked)
        return out;

    // Masked heads: attention over all of the slot's keys with the text keys outside the window blocked.
    const int M = io.masked, cap = g.slot_cap;
    ggml_tensor* qm = ggml_get_rows(ctx, ggml_reshape_2d(ctx, q4, (int64_t)hd_kv * nh, W), io.pm_rows);
    qm = ggml_reshape_4d(ctx, qm, hd_kv, 1, nh, M);
    ggml_tensor* km = ggml_reshape_4d(ctx, ggml_get_rows(ctx, kc, io.pm_kidx), hd_kv, cap, M, nkv);
    km = ggml_cont(ctx, ggml_permute(ctx, km, 0, 1, 3, 2)); // [hd_kv, cap, nkv, M]
    ggml_tensor* vm = ggml_reshape_4d(ctx, ggml_get_rows(ctx, vc, io.pm_kidx), hd_kv, cap, M, nkv);
    vm = ggml_cont(ctx, ggml_permute(ctx, vm, 1, 0, 3, 2)); // [cap, hd_kv, nkv, M]
    ggml_tensor* sc = matmul(ctx, km, qm);                  // [cap, 1, nh, M]
    ggml_tensor* kpos = ggml_repeat(ctx, io.pm_kpos, sc);
    ggml_tensor* tidx = ggml_sub(ctx, kpos, io.pm_ts);
    ggml_tensor* in_text =
        ggml_mul(ctx, ge0(tidx), le0(ggml_scale_bias(ctx, ggml_sub(ctx, tidx, io.pm_tl), 1.0f, 1.0f)));
    ggml_tensor* in_win = ggml_mul(ctx, ge0(ggml_sub(ctx, tidx, io.pm_lo[s])), le0(ggml_sub(ctx, tidx, io.pm_hi[s])));
    ggml_tensor* blocked = ggml_mul(ctx, in_text, ggml_scale_bias(ctx, in_win, -1.0f, 1.0f));
    ggml_tensor* keep =
        ggml_mul(ctx, le0(ggml_sub(ctx, kpos, io.pm_npast)), ggml_scale_bias(ctx, blocked, -1.0f, 1.0f));
    ggml_tensor* pm = ggml_soft_max_ext(ctx, sc, additive(keep), scale, 0.0f);
    ggml_tensor* om = ggml_reshape_3d(ctx, matmul(ctx, vm, pm), hd_kv, nh, M); // [hd_kv, nh, M]
    if (hd_kv > hd)
        om = ggml_cont(ctx, ggml_view_3d(ctx, om, hd, nh, M, om->nb[1], om->nb[2], 0));
    ggml_tensor* rows = ggml_reshape_2d(ctx, out, (int64_t)hd * nh, W);
    ggml_tensor* old = ggml_reshape_3d(ctx, ggml_get_rows(ctx, rows, io.pm_rows), hd, nh, M);
    ggml_tensor* flag = io.pm_flag[s];
    // flag 1: 0*old + om == om; flag 0: old + 0*om == old (bit-exact selection).
    ggml_tensor* mixed =
        ggml_add(ctx, ggml_mul(ctx, old, ggml_scale_bias(ctx, flag, -1.0f, 1.0f)), ggml_mul(ctx, om, flag));
    rows = ggml_set_rows(ctx, rows, ggml_reshape_2d(ctx, mixed, (int64_t)hd * nh, M), io.pm_rows);
    return ggml_reshape_3d(ctx, rows, hd, nh, W);
}

// Attention block: gater sigmoid, QK-RMSNorm + per-head temp, interleaved RoPE,
// GQA causal attention (scale 1/sqrt(head_dim)), per-head sigmoid gate, wo.
// When g.kv: write the new tokens' K/V into the per-layer cache at [pos0, pos0+n) and
// attend against the full cache [0, pos0+n). Otherwise self-attend over the n tokens.
static ggml_tensor* build_attention(gctx& g, ggml_tensor* cur, const zonos3_layer& ly, int il) {
    const auto& hp = g.m->hp;
    ggml_context* ctx = g.ctx;
    const int hd = (int)hp.head_dim, nh = (int)hp.n_head, nkv = (int)hp.n_head_kv;
    const int hd_kv = zonos3_kv_head_dim(hp), n = g.n;
    const std::string blk = "blk." + std::to_string(il) + ".";
    ggml_tensor* gate = ggml_sigmoid(ctx, lin(g, ly.gater, cur, blk + "attn_gate"));
    ggml_tensor* q = ggml_reshape_3d(ctx, lin(g, ly.wq, cur, blk + "attn_q"), hd, nh, n);
    ggml_tensor* k = ggml_reshape_3d(ctx, lin(g, ly.wk, cur, blk + "attn_k"), hd, nkv, n);
    ggml_tensor* v = ggml_reshape_3d(ctx, lin(g, ly.wv, cur, blk + "attn_v"), hd, nkv, n);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.qk_norm_eps), ggml_reshape_3d(ctx, ly.attn_temp, 1, nh, 1));
    k = ggml_rms_norm(ctx, k, hp.qk_norm_eps);
    q = ggml_rope_ext(ctx, q, g.pos_rope, nullptr, hd, GGML_ROPE_TYPE_NORMAL, (int)hp.n_ctx_train, hp.rope_freq_base, 1,
                      0, 1, 32, 1);
    k = ggml_rope_ext(ctx, k, g.pos_rope, nullptr, hd, GGML_ROPE_TYPE_NORMAL, (int)hp.n_ctx_train, hp.rope_freq_base, 1,
                      0, 1, 32, 1);
    ggml_tensor* out = nullptr;
    if (g.kc) {
        ggml_tensor *kc = (*g.kc)[il], *vc = (*g.vc)[il];
        ggml_tensor* kb = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor* vb = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        if (hd_kv > hd) {
            kb = ggml_pad(ctx, kb, hd_kv - hd, 0, 0, 0);
            vb = ggml_pad(ctx, vb, hd_kv - hd, 0, 0, 0);
        }
        ggml_build_forward_expand(g.gf, ggml_set_rows(ctx, kc, kb, g.pos_cache));
        ggml_build_forward_expand(g.gf, ggml_set_rows(ctx, vc, vb, g.pos_cache));
        if (g.decode) {
            // Each band has a private extra row for idle writes. It is never
            // attended, so an inactive, already-prefilled slot remains intact.
            const size_t band = (size_t)(g.slot_cap + 1) * kc->nb[1];
            ggml_tensor* k4 = ggml_view_4d(ctx, kc, hd_kv, g.slot_cap, nkv, g.n_slots, kc->nb[1], kc->nb[2], band, 0);
            ggml_tensor* v4 = ggml_view_4d(ctx, vc, hd_kv, g.slot_cap, nkv, g.n_slots, vc->nb[1], vc->nb[2], band, 0);
            ggml_tensor* q4 = ggml_reshape_4d(ctx, q, hd, 1, nh, g.n_slots);
            if (hd_kv > hd)
                q4 = ggml_pad(ctx, q4, hd_kv - hd, 0, 0, 0);
            if (kc->type == GGML_TYPE_F32) {
                // Full-F32 diagnostics must not lose precision inside CUDA
                // FlashAttention, which converts even F32 K/V to F16.
                ggml_tensor* scores = matmul(ctx, k4, q4);
                scores = ggml_soft_max_ext(ctx, scores, g.mask, 1.0f / sqrtf((float)hd), 0);
                ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v4, 1, 0, 2, 3));
                out = ggml_cont(ctx, ggml_permute(ctx, matmul(ctx, vp, scores), 0, 2, 1, 3));
            } else {
                // CUDA dispatches isolated queries to VEC but multi-sequence
                // queries to MMA, which rounds Q and softmax weights to F16.
                // Keep each request's attention arithmetic independent of the
                // active batch width, retaining batched projections/GDN/MoE.
                std::vector<ggml_tensor*> attended_slots;
                attended_slots.reserve(g.n_slots);
                for (int slot = 0; slot < g.n_slots; ++slot) {
                    ggml_tensor* qs = ggml_view_4d(ctx, q4, hd_kv, 1, nh, 1, q4->nb[1], q4->nb[2], q4->nb[3],
                                                   (size_t)slot * q4->nb[3]);
                    ggml_tensor* ks = ggml_view_4d(ctx, k4, hd_kv, g.slot_cap, nkv, 1, k4->nb[1], k4->nb[2], k4->nb[3],
                                                   (size_t)slot * k4->nb[3]);
                    ggml_tensor* vs = ggml_view_4d(ctx, v4, hd_kv, g.slot_cap, nkv, 1, v4->nb[1], v4->nb[2], v4->nb[3],
                                                   (size_t)slot * v4->nb[3]);
                    ggml_tensor* mask = ggml_view_4d(ctx, g.mask, g.slot_cap, 1, 1, 1, g.mask->nb[1], g.mask->nb[2],
                                                     g.mask->nb[3], (size_t)slot * g.mask->nb[3]);
                    attended_slots.push_back(ggml_flash_attn_ext(ctx, qs, ks, vs, mask, 1.0f / sqrtf((float)hd), 0, 0));
                }
                // Balanced concatenation avoids quadratic copying at large widths.
                for (size_t count = attended_slots.size(); count > 1;) {
                    size_t next = 0;
                    for (size_t i = 0; i < count; i += 2)
                        attended_slots[next++] = i + 1 < count
                                                     ? ggml_concat(ctx, attended_slots[i], attended_slots[i + 1], 3)
                                                     : attended_slots[i];
                    count = next;
                }
                out = attended_slots.front();
            }
            out = ggml_reshape_3d(ctx, out, hd_kv, nh, g.n_slots);
            if (hd_kv > hd)
                out = ggml_cont(ctx, ggml_view_3d(ctx, out, hd, nh, g.n_slots, out->nb[1], out->nb[2], 0));
            if (g.io && g.io->lmax)
                out = build_probe(g, q4, kc, vc, out, il);
        } else {
            // Per segment: its queries against its own band's cache rows [0, position + n_s).
            const auto segs = g.segments();
            for (size_t s = 0; s < segs.size(); ++s) {
                const auto& sg = segs[s];
                ggml_tensor* qs = segs.size() == 1 ? q
                                                   : ggml_view_3d(ctx, q, hd, nh, sg.n, q->nb[1], q->nb[2],
                                                                  (size_t)sg.row0 * q->nb[2]);
                const int keys = sg.position + sg.n;
                const size_t offset = (size_t)sg.cache_slot * (g.slot_cap + 1) * kc->nb[1];
                ggml_tensor* ks = ggml_view_3d(ctx, kc, hd, keys, nkv, kc->nb[1], kc->nb[2], offset);
                ggml_tensor* vs = ggml_view_3d(ctx, vc, hd, keys, nkv, vc->nb[1], vc->nb[2], offset);
                // Cached keys already have RoPE, head normalization and query scale.
                ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, qs, 0, 2, 1, 3));
                ggml_tensor* kp = ggml_cont(ctx, ks);
                ggml_tensor* scores = matmul(ctx, kp, qp);
                scores = ggml_soft_max_ext(ctx, scores, g.masks_causal[s], 1.0f / sqrtf((float)hd), 0);
                ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, vs, 1, 0, 2, 3));
                ggml_tensor* os = ggml_cont(ctx, ggml_permute(ctx, matmul(ctx, vp, scores), 0, 2, 1, 3));
                out = out ? ggml_concat(ctx, out, os, 2) : os;
            }
        }
    } else {
        ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor* scores = ggml_soft_max_ext(ctx, matmul(ctx, kp, qp), g.mask_causal, 1.0f / sqrtf((float)hd), 0);
        ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        out = ggml_cont(ctx, ggml_permute(ctx, matmul(ctx, vp, scores), 0, 2, 1, 3));
    }
    out = ggml_mul(ctx, out, ggml_reshape_3d(ctx, gate, 1, nh, n));
    return lin(g, ly.wo, ggml_reshape_2d(ctx, out, hd * nh, n), blk + "attn_output");
}

static ggml_tensor* build_gdn(gctx& g, ggml_tensor* cur, const zonos3_layer& ly, int L) {
    ggml_context* ctx = g.ctx;
    const auto& hp = g.m->hp;
    const int H = (int)hp.gdn_heads, S = (int)hp.gdn_head_dim, I = H * S;
    const int R = (int)hp.gdn_conv_kernel - 1, B = g.decode ? g.n_slots : 1;
    const std::string suffix = "_" + std::to_string(L);
    ggml_tensor* mixed = lin(g, ly.gdn_in, cur, "blk." + std::to_string(L) + ".gdn_in");
    ggml_tensor* z =
        ggml_view_3d(ctx, mixed, S, H, g.n, (size_t)S * mixed->nb[0], mixed->nb[1], (size_t)3 * I * mixed->nb[0]);
    // Short conv + gated delta rule over `nr` rows of `mx` (one sequence of a prefill, or the decode batch) with the
    // state of cache slot `slot`; returns the recurrent output [I, nr].
    auto recurrent = [&](ggml_tensor* mx, int nr, size_t slot) {
        const int T = g.decode ? 1 : nr;
        // Column slices of `mx` are fed as strided views: every consumer (concat, rms_norm,
        // add, sigmoid, silu) only needs contiguous rows on CPU and Metal, so no copies are made.
        auto rows = [&](ggml_tensor* x, int first, int count) {
            return ggml_view_2d(ctx, x, count, nr, x->nb[1], (size_t)first * x->nb[0]);
        };
        ggml_tensor* raw = rows(mx, 0, 3 * I);
        ggml_tensor *a = rows(mx, 4 * I, H), *b = rows(mx, 4 * I + H, H);
        ggml_tensor *history, *state;
        if (g.state) {
            ggml_tensor *hs = (*g.history)[L], *ss = (*g.state)[L];
            history = ggml_view_3d(ctx, hs, R, 3 * I, B, hs->nb[1], hs->nb[2], slot * hs->nb[2]);
            state = ggml_view_4d(ctx, ss, S, S, H, B, ss->nb[1], ss->nb[2], ss->nb[3], slot * ss->nb[3]);
        } else {
            history = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, R, 3 * I, B);
            state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);
            ggml_set_input(history);
            ggml_set_input(state);
            g.zero_inputs.push_back(history);
            g.zero_inputs.push_back(state);
        }
        ggml_tensor* time_major = g.decode
                                      ? ggml_view_3d(ctx, mx, 1, 3 * I, B, mx->nb[0], mx->nb[1], 0)
                                      : ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, raw)), T, 3 * I, 1);
        ggml_tensor* window = ggml_concat(ctx, history, time_major, 0);
        ggml_tensor* conv = ggml_silu(ctx, ggml_ssm_conv(ctx, window, ly.gdn_conv));
        conv = ggml_reshape_2d(ctx, conv, 3 * I, nr);
        g.cap("gdn_conv" + suffix, conv);
        // GATED_DELTA_NET reads q/k/v through their strides (CPU needs only contiguous rows).
        auto heads = [&](int first) {
            return ggml_view_4d(ctx, conv, S, H, T, B, (size_t)S * conv->nb[0], conv->nb[1], (size_t)T * conv->nb[1],
                                (size_t)first * conv->nb[0]);
        };
        ggml_tensor *q = heads(0), *k = heads(I), *v = heads(2 * I);
        // FLA uses sqrt(sum(x*x)+eps), unlike ggml_l2_norm's max(norm,eps).
        auto l2 = [&](ggml_tensor* x) {
            return ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f / S), 1.0f / sqrtf((float)S));
        };
        q = l2(q);
        k = l2(k);
        g.cap("gdn_q" + suffix, ggml_reshape_3d(ctx, q, S, H, nr));
        g.cap("gdn_k" + suffix, ggml_reshape_3d(ctx, k, S, H, nr));
        g.cap("gdn_v" + suffix,
              ggml_view_3d(ctx, conv, S, H, nr, (size_t)S * conv->nb[0], conv->nb[1], (size_t)2 * I * conv->nb[0]));
        ggml_tensor* gate = ggml_neg(
            ctx, ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, a, ly.gdn_dt_bias)), ggml_exp(ctx, ly.gdn_A_log)));
        ggml_tensor* beta = ggml_sigmoid(ctx, b);
        if (g.decode) {
            ggml_tensor* active = ggml_reshape_2d(ctx, g.active, 1, B);
            gate = ggml_mul(ctx, gate, active);
            beta = ggml_mul(ctx, beta, active);
        }
        g.cap("gdn_decay" + suffix, ggml_exp(ctx, gate));
        g.cap("gdn_beta" + suffix, beta);
        gate = ggml_reshape_4d(ctx, gate, 1, H, T, B);
        beta = ggml_reshape_4d(ctx, beta, 1, H, T, B);
        ggml_tensor* packed = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, 1);
        ggml_tensor* output = ggml_view_2d(ctx, packed, I, nr, (size_t)I * sizeof(float), 0);
        g.cap("gdn_recurrent" + suffix, ggml_reshape_3d(ctx, output, S, H, nr));
        const size_t state_offset = (size_t)I * nr * sizeof(float);
        ggml_tensor* next_state =
            ggml_view_4d(ctx, packed, S, S, H, B, (size_t)S * sizeof(float), (size_t)S * S * sizeof(float),
                         (size_t)S * S * H * sizeof(float), state_offset);
        ggml_tensor* next_history = ggml_cont(
            ctx, ggml_view_3d(ctx, window, R, 3 * I, B, window->nb[1], window->nb[2], (size_t)T * window->nb[0]));
        if (g.decode) {
            ggml_tensor* active = ggml_reshape_3d(ctx, g.active, 1, 1, B);
            ggml_tensor* inactive = ggml_scale_bias(ctx, active, -1.0f, 1.0f);
            next_history = ggml_add(ctx, ggml_mul(ctx, next_history, active), ggml_mul(ctx, history, inactive));
        }
        g.cap("gdn_state" + suffix, next_state);
        g.cap("gdn_conv_state" + suffix, next_history);
        if (g.state) {
            g.updates.emplace_back(next_state, state);
            g.updates.emplace_back(next_history, history);
        }
        return output;
    };
    ggml_tensor* output = nullptr;
    if (g.decode) {
        output = recurrent(mixed, g.n, 0);
    } else {
        const auto segs = g.segments();
        for (const auto& sg : segs) {
            ggml_tensor* mx = segs.size() == 1 ? mixed
                                               : ggml_view_2d(ctx, mixed, mixed->ne[0], sg.n, mixed->nb[1],
                                                              (size_t)sg.row0 * mixed->nb[1]);
            ggml_tensor* os = recurrent(mx, sg.n, (size_t)sg.cache_slot);
            output = output ? ggml_concat(ctx, output, os, 1) : os;
        }
    }
    output = ggml_reshape_3d(ctx, output, S, H, g.n);
    output = ggml_mul(ctx, rms_w(ctx, output, ly.gdn_norm, 1e-5f), ggml_reshape_3d(ctx, ggml_silu(ctx, z), S, H, g.n));
    output = g.cap("gdn_gated" + suffix, ggml_reshape_2d(ctx, output, I, g.n));
    return lin(g, ly.wo, output, "blk." + std::to_string(L) + ".attn_output");
}
// Dense SwiGLU FFN: down( up(x) * silu(gate(x)) ).  up=chunk0, gate=chunk1 (silu on gate).
static ggml_tensor* build_dense_ffn(gctx& g, ggml_tensor* cur, const zonos3_layer& ly, int L) {
    ggml_context* ctx = g.ctx;
    const std::string blk = "blk." + std::to_string(L) + ".";
    ggml_tensor* up = lin(g, ly.ffn_up, cur, blk + "ffn_up");
    ggml_tensor* gate = lin(g, ly.ffn_gate, cur, blk + "ffn_gate");
    ggml_tensor* y = ggml_mul(ctx, up, ggml_silu(ctx, gate));
    return lin(g, ly.ffn_down, y, blk + "ffn_down");
}

// Latent MoE block with a linear router (weighted RMSNorm -> linear, no EDA).
// Quantile selection on logits−bias; weights = softmax over the k selected logits. Experts run
// in the latent space: z = latent_down·x, silu(gate.z)*(up.z) -> down, weighted sum over top-k,
// then latent_up back to n_embd.
static ggml_tensor* build_moe(gctx& g, ggml_tensor* cur, const zonos3_layer& ly, int L) {
    const zonos3_hparams& hp = g.m->hp;
    ggml_context* ctx = g.ctx;
    // Per-layer expert count (not hp.n_expert): a pruned GGUF can keep a different number of
    // experts per layer, so derive it from the actual stack. The router/router_bias the
    // softmax runs over are sliced to match, so softmax renormalizes over the survivors.
    const int ne = (int)ly.ffn_up_exps->ne[2], k = ly.top_k, n = g.n;
    const int n_latent = (int)ly.latent_down->ne[1];
    const std::string blk = "blk." + std::to_string(L) + ".";

    ggml_tensor* rn = rms_w(ctx, cur, ly.router_norm, hp.rms_eps);
    ggml_tensor* rlogits = lin(g, ly.router_w, rn, blk + "router"); // [ne, n]
    // A runtime mask affects selection and the selected raw logits, not balancing biases.
    if (ly.router_mask)
        rlogits = ggml_add(ctx, rlogits, ly.router_mask);
    g.cap("router_logits_" + std::to_string(L), rlogits);
    ggml_tensor* bias = ly.router_bias;
    if (g.io && g.io->accent) {
        // Accent router shift per row: bias + strength * shift (rows without an accent add an exact zero).
        const auto it = g.io->ac_rbias.find(L);
        if (it != g.io->ac_rbias.end())
            bias = ggml_add(ctx, it->second, ly.router_bias);
    }
    ggml_tensor* scores = ggml_sub(ctx, rlogits, bias);
    if (ly.router_mask)
        scores = ggml_add(ctx, scores, ly.router_mask);
    ggml_tensor* sel = ggml_top_k(ctx, scores, k);
    ggml_tensor* selected = ggml_get_rows(ctx, ggml_reshape_3d(ctx, rlogits, 1, ne, n), sel);
    ggml_tensor* weights = ggml_reshape_3d(ctx, ggml_soft_max(ctx, ggml_reshape_2d(ctx, selected, k, n)), 1, k, n);

    // Accent expert factors: per routed pair, shift += up_e @ ((down_e @ row) * strength) over the stacked adapters.
    auto expert_shift = [&](const char* stack, ggml_tensor* rows) -> ggml_tensor* {
        if (!g.io || !g.io->accent || !g.io->ac_emask)
            return nullptr;
        const auto it = g.m->accents->experts.find(blk + stack);
        if (it == g.m->accents->experts.end())
            return nullptr;
        ggml_tensor* t = ggml_mul(ctx, matmul_id(ctx, it->second.down, rows, sel), g.io->ac_emask);
        return matmul_id(ctx, it->second.up, t, sel);
    };
    ggml_tensor* z = lin(g, ly.latent_down, cur, blk + "ffn_latent_down"); // [n_latent, n]
    ggml_tensor* z3 = ggml_reshape_3d(ctx, z, n_latent, 1, n);
    ggml_tensor* up = matmul_id(ctx, ly.ffn_up_exps, z3, sel);     // [n_ff, k, n]
    ggml_tensor* gate = matmul_id(ctx, ly.ffn_gate_exps, z3, sel); // [n_ff, k, n]
    if (ggml_tensor* shift = expert_shift("ffn_up_exps", z3))
        up = ggml_add(ctx, up, shift);
    if (ggml_tensor* shift = expert_shift("ffn_gate_exps", z3))
        gate = ggml_add(ctx, gate, shift);
    ggml_tensor* y = ggml_mul(ctx, up, ggml_silu(ctx, gate));
    if (g.cap_moe) {
        g.out_named("moe_in_" + std::to_string(L), z);    // [n_latent, n] gate/up input
        g.out_named("moe_y_" + std::to_string(L), y);     // [n_ff, k, n] down input
        g.out_named("moe_sel_" + std::to_string(L), sel); // [k, n] i32   routing
    }
    ggml_tensor* exps = matmul_id(ctx, ly.ffn_down_exps, y, sel); // [n_latent, k, n]
    if (ggml_tensor* shift = expert_shift("ffn_down_exps", y))
        exps = ggml_add(ctx, exps, shift);
    exps = ggml_mul(ctx, exps, weights); // scale by route prob

    ggml_tensor* moe = ggml_view_2d(ctx, exps, n_latent, n, exps->nb[2], 0);
    for (int j = 1; j < k; ++j) {
        moe = ggml_add(ctx, moe, ggml_view_2d(ctx, exps, n_latent, n, exps->nb[2], (size_t)j * exps->nb[1]));
    }
    return lin(g, ly.latent_up, ggml_cont(ctx, moe), blk + "ffn_latent_up"); // [n_embd, n]
}

// Which replaced rows of segment `s` fall inside its chunk [position, position+n)?
static bool chunk_has_speaker(const gctx& g, const gctx::segment& s) {
    return !g.decode && s.cond && s.cond->speaker_row >= s.position && s.cond->speaker_row < s.position + s.n;
}
static bool chunk_has_condition(const gctx& g, const gctx::segment& s) {
    if (g.decode || !s.cond || s.cond->condition_row < 0)
        return false;
    return s.cond->condition_row < s.position + s.n && s.cond->condition_row + (int)g.m->hp.n_cond > s.position;
}

// Build the prefill graph. Returns the final node. n_layer_limit < 0 => all layers.
ggml_tensor* build_graph(gctx& g, int n_layer_limit) {
    const auto& m = *g.m;
    const auto& hp = m.hp;
    ggml_context* ctx = g.ctx;
    const int W = (int)hp.n_codebooks + 1, D = (int)hp.n_embd;
    g.ids.resize(W);
    ggml_tensor* emb = nullptr;
    for (int c = 0; c < W; ++c) {
        ggml_tensor* id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, g.n);
        ggml_set_input(id);
        ggml_set_name(id, ("ids." + std::to_string(c)).c_str());
        g.ids[c] = id;
        ggml_tensor* table = c < (int)hp.n_codebooks ? m.audio_embd[c] : m.text_embd;
        ggml_tensor* e = ggml_get_rows(ctx, table, id);
        emb = emb ? ggml_add(ctx, emb, e) : e;
    }
    g.cap("emb_sum", emb);
    const auto segs = g.segments();
    bool any_replaced = false;
    for (const auto& s : segs)
        any_replaced = any_replaced || chunk_has_speaker(g, s) || chunk_has_condition(g, s);
    if (any_replaced) {
        // Replace the summed rows (before emb_norm) with speaker_proj(speaker) / cond_embd[i][bucket_i]
        // through functional ops: emb = emb * keep + table[sel]. Column 0 of the table is zero.
        g.ovr_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, g.n);
        g.ovr_sel = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, g.n);
        g.ovr_zero = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
        for (ggml_tensor* t : {g.ovr_keep, g.ovr_sel, g.ovr_zero})
            ggml_set_input(t);
        g.zero_inputs.push_back(g.ovr_zero);
        ggml_tensor* table = g.ovr_zero;
        int columns = 1;
        g.ovr_spk_col.assign(segs.size(), -1);
        g.ovr_cond_col.assign(segs.size(), -1);
        for (size_t s = 0; s < segs.size(); ++s) {
            if (!chunk_has_speaker(g, segs[s]))
                continue;
            ggml_tensor* spk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)hp.speaker_dim, 1);
            ggml_set_input(spk);
            g.ovr_spks.push_back(spk);
            ggml_tensor* projected = ggml_add(ctx, matmul(ctx, m.speaker_proj_w, spk), m.speaker_proj_b);
            table = ggml_concat(ctx, table, projected, 1);
            g.ovr_spk_col[s] = columns++;
        }
        for (size_t s = 0; s < segs.size(); ++s) {
            if (!chunk_has_condition(g, segs[s]))
                continue;
            ggml_tensor* bkt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)hp.n_cond);
            ggml_set_input(bkt);
            g.ovr_bkts.push_back(bkt);
            for (uint32_t i = 0; i < hp.n_cond; ++i) {
                ggml_tensor* bucket = ggml_view_1d(ctx, bkt, 1, (size_t)i * sizeof(int32_t));
                table = ggml_concat(ctx, table, ggml_get_rows(ctx, m.cond_embd[i], bucket), 1);
            }
            g.ovr_cond_col[s] = columns;
            columns += (int)hp.n_cond;
        }
        emb = ggml_add(ctx, ggml_mul(ctx, emb, g.ovr_keep), ggml_get_rows(ctx, table, g.ovr_sel));
    }
    g.cap("emb_cond", emb);
    ggml_tensor* x = g.cap("emb_norm", ggml_rms_norm(ctx, emb, hp.rms_eps));
    const int nl = n_layer_limit < 0 ? (int)hp.n_layer : std::min(n_layer_limit, (int)hp.n_layer);
    if (nl > 0) {
        g.pos_rope = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, g.n);
        ggml_set_input(g.pos_rope);
        if (g.kc) {
            g.pos_cache = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, g.n);
            ggml_set_input(g.pos_cache);
        }
        if (!g.decode && nl >= 4) {
            for (const auto& s : segs) {
                g.masks_causal.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, s.position + s.n, s.n));
                ggml_set_input(g.masks_causal.back());
            }
            g.mask_causal = g.masks_causal[0];
        }
    }
    if (g.decode) {
        g.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, g.slot_cap, 1, 1, g.n_slots);
        g.active = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, g.n_slots);
        ggml_set_input(g.mask);
        ggml_set_input(g.active);
    }
    if (g.decode) {
        for (size_t j = 0; j < g.steer_sources.size(); ++j) {
            ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, g.n);
            ggml_set_input(t);
            ggml_set_name(t, ("steer." + std::to_string(g.steer_sources[j])).c_str());
            g.steer.push_back(t);
        }
    }
    bool any_accent = false;
    for (const auto& s : segs)
        any_accent = any_accent || (s.cond && accent_on(m, s.cond->accent, s.cond->accent_strength));
    if (!g.decode && any_accent) {
        if (!g.io)
            g.io = std::make_shared<zonos3_feature_io>();
        g.io->accent = true;
    }
    if (g.io && g.io->accent) {
        const zonos3_accents& bank = *m.accents;
        auto input = [&](ggml_tensor* t, const std::string& name) {
            ggml_set_input(t);
            ggml_set_name(t, name.c_str());
            return t;
        };
        if (bank.K)
            g.io->ac_mask = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)bank.n * bank.K, g.n), "accent.mask");
        if (bank.R)
            g.io->ac_emask =
                input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int64_t)bank.n * bank.R, 1, g.n), "accent.emask");
        for (const auto& rb : bank.router_bias)
            g.io->ac_rbias[rb.first] =
                input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.layers[rb.first].router_bias->ne[0], g.n),
                      "accent.router_bias." + std::to_string(rb.first));
    }
    if (g.decode && g.io && g.io->lmax) {
        auto& io = *g.io;
        const int64_t nh = hp.n_head, nkv = hp.n_head_kv, Wd = g.n_slots, M = io.masked;
        auto input = [&](ggml_tensor* t) {
            ggml_set_input(t);
            return t;
        };
        io.pr_kidx = input(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, (int64_t)io.lmax * Wd, nkv));
        io.pr_tpos = input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, io.lmax, 1, 1, 1));
        if (M) {
            io.pm_rows = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, M));
            io.pm_kidx = input(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, M * g.slot_cap, nkv));
            io.pm_kpos = input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g.slot_cap, 1, 1, 1));
            io.pm_npast = input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, M));
            io.pm_ts = input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, M));
            io.pm_tl = input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, M));
        }
        for (size_t s = 0; s < io.attn_layers.size(); ++s) {
            io.pr_w.push_back(input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nh, 1, Wd)));
            io.pr_lo.push_back(input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, nh, Wd)));
            io.pr_hi.push_back(input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, nh, Wd)));
            if (!M)
                continue;
            io.pm_flag.push_back(input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, nh, M)));
            io.pm_lo.push_back(input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, nh, M)));
            io.pm_hi.push_back(input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, nh, M)));
        }
    }
    // Full AttnRes routing over the s sources seen so far. Each new source is scored once
    // against every query row (Q = 2*n_layer+1) and appended to two running stacks:
    //   SC  [Q, s, n] key scores,  SRC [D, s, n] sources.
    // A route takes one softmax over its score row and one broadcast multiply; the weighted
    // sources are then summed in source order with plain adds, which keeps the reference
    // F32 rounding (a dot-product matmul over s would reorder it). Metal fuses the add chain.
    const int Q = (int)m.attnres_q->ne[1];
    ggml_tensor *SC = nullptr, *SRC = nullptr;
    int n_src = 0;
    auto route = [&](ggml_tensor* source, int qi) {
        if (!g.steer.empty()) {
            // Hidden-space steering: the delta joins the source before it enters the bank.
            const auto it = std::find(g.steer_sources.begin(), g.steer_sources.end(), qi);
            if (it != g.steer_sources.end())
                source = ggml_add(ctx, source, g.steer[it - g.steer_sources.begin()]);
        }
        ggml_tensor* sc = matmul(ctx, m.attnres_q, ggml_rms_norm(ctx, source, hp.rms_eps)); // [Q, n]
        sc = ggml_reshape_3d(ctx, sc, Q, 1, g.n);
        ggml_tensor* col = ggml_reshape_3d(ctx, source, D, 1, g.n);
        SC = SC ? ggml_concat(ctx, SC, sc, 1) : sc;
        SRC = SRC ? ggml_concat(ctx, SRC, col, 1) : col;
        if (++n_src == 1)
            return source;
        ggml_tensor* row = ggml_view_3d(ctx, SC, 1, n_src, g.n, SC->nb[1], SC->nb[2], (size_t)qi * SC->nb[0]);
        ggml_tensor* w = ggml_soft_max(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, row), n_src, g.n));
        ggml_tensor* weighted = ggml_mul(ctx, SRC, ggml_reshape_3d(ctx, w, 1, n_src, g.n)); // [D, s, n]
        auto slice = [&](int s) {
            return ggml_view_2d(ctx, weighted, D, g.n, weighted->nb[2], (size_t)s * weighted->nb[1]);
        };
        ggml_tensor* mix = slice(0);
        for (int s = 1; s < n_src; ++s)
            mix = ggml_add(ctx, mix, slice(s));
        return mix;
    };
    for (int L = 0; L < nl; ++L) {
        if (L < (int)m.layer_skip.size() && m.layer_skip[L])
            continue;
        const auto& ly = m.layers[L];
        const std::string suffix = "_" + std::to_string(L);
        ggml_tensor* mixed = g.cap("attnres_attn" + suffix, route(x, 2 * L));
        ggml_tensor* input = g.cap("attn_in" + suffix, rms_w(ctx, mixed, ly.attn_norm, hp.rms_eps));
        ggml_tensor* attention =
            g.cap("attn_out" + suffix, ly.is_gdn ? build_gdn(g, input, ly, L) : build_attention(g, input, ly, L));
        mixed = g.cap("attnres_ffn" + suffix, route(attention, 2 * L + 1));
        input = g.cap("ffn_in" + suffix, rms_w(ctx, mixed, ly.ffn_norm, hp.rms_eps));
        x = g.cap("ffn_out" + suffix, ly.is_moe ? build_moe(g, input, ly, L) : build_dense_ffn(g, input, ly, L));
        g.cap("layer_x" + suffix, x);
    }
    if (n_layer_limit < 0) {
        ggml_tensor* mixed = g.cap("attnres_final", route(x, 2 * (int)hp.n_layer));
        ggml_tensor* h = g.cap("out_norm", rms_w(ctx, mixed, m.output_norm, hp.rms_eps));
        ggml_tensor* head = g.cap("mout", lin(g, m.output, h, "output"));
        x = ggml_reshape_3d(ctx, head, hp.audio_vocab, hp.n_codebooks, g.n);
        if (hp.logit_softcap > 0)
            x = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, x, 1.0f / hp.logit_softcap)), hp.logit_softcap);
        g.cap("logits", x);
    }
    if (g.io && g.io->lmax && g.probe_acc) {
        g.io->pr_acc = ggml_cont(ctx, g.probe_acc); // own buffer: a view marked output does not pin its base
        ggml_set_output(g.io->pr_acc);
        g.extra_outputs.push_back(g.io->pr_acc);
    }
    return x;
}

bool save_tensor(const std::string& dir, const std::string& name, ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "validate: skip %s (type %s)\n", name.c_str(), ggml_type_name(t->type));
        return true;
    }
    const int nd = ggml_n_dims(t);
    std::vector<int64_t> shape(nd > 0 ? nd : 1, 1);
    for (int i = 0; i < nd; ++i)
        shape[i] = t->ne[nd - 1 - i]; // reverse ne -> numpy C-order
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    return npy::save_f32(dir + "/" + name + ".npy", buf.data(), shape);
}

static bool valid_ids(const zonos3_model& m, const float* ids, int n) {
    if (!ids || n <= 0 || n > (int)m.hp.n_ctx_train)
        return false;
    const int W = (int)m.hp.n_codebooks + 1;
    for (int t = 0; t < n; ++t)
        for (int c = 0; c < W; ++c) {
            const float v = ids[(size_t)t * W + c];
            const uint32_t limit = c < W - 1 ? m.hp.audio_vocab : m.hp.text_vocab + 1 + m.hp.text_extra_tokens;
            if (!std::isfinite(v) || v < 0 || v >= limit || v != std::floor(v)) {
                fprintf(stderr, "zonos3: invalid token at position %d column %d\n", t, c);
                return false;
            }
        }
    return true;
}

// Accent inputs of `cols` graph rows; sel[c] = (adapter, strength), adapter < 0 = none (exact zero shift).
static void set_accent_inputs(const zonos3_model& m, const zonos3_feature_io& io, int cols,
                              const std::vector<std::pair<int, float>>& sel) {
    const zonos3_accents& bank = *m.accents;
    auto fill_mask = [&](ggml_tensor* t, int per) {
        if (!t)
            return;
        std::vector<float> h((size_t)bank.n * per * cols, 0.0f);
        for (int c = 0; c < cols; ++c)
            if (sel[c].first >= 0)
                std::fill_n(h.begin() + ((size_t)c * bank.n + sel[c].first) * per, per, sel[c].second);
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(float));
    };
    fill_mask(io.ac_mask, bank.K);
    fill_mask(io.ac_emask, bank.R);
    for (const auto& rb : io.ac_rbias) {
        const auto& shifts = bank.router_bias.at(rb.first);
        const size_t E = (size_t)rb.second->ne[0];
        std::vector<float> h(E * cols, 0.0f);
        for (int c = 0; c < cols; ++c) {
            const int a = sel[c].first;
            if (a < 0 || shifts[a].empty())
                continue;
            for (size_t e = 0; e < E; ++e)
                h[c * E + e] = sel[c].second * shifts[a][e]; // strength * shift (fp32)
        }
        ggml_backend_tensor_set(rb.second, h.data(), 0, h.size() * sizeof(float));
    }
}

static void set_graph_inputs(gctx& g, const float* ids) {
    const int W = (int)g.m->hp.n_codebooks + 1;
    std::vector<int32_t> column(g.n);
    for (int c = 0; c < W; ++c) {
        for (int t = 0; t < g.n; ++t)
            column[t] = (int32_t)ids[(size_t)t * W + c];
        ggml_backend_tensor_set(g.ids[c], column.data(), 0, column.size() * sizeof(int32_t));
    }
    const auto segs = g.segments();
    if (g.pos_rope) {
        for (const auto& s : segs)
            for (int t = 0; t < s.n; ++t)
                column[s.row0 + t] = s.position + t;
        ggml_backend_tensor_set(g.pos_rope, column.data(), 0, column.size() * sizeof(int32_t));
    }
    if (g.pos_cache) {
        for (const auto& s : segs)
            for (int t = 0; t < s.n; ++t)
                column[s.row0 + t] = s.cache_slot * (g.slot_cap + 1) + s.position + t;
        ggml_backend_tensor_set(g.pos_cache, column.data(), 0, column.size() * sizeof(int32_t));
    }
    for (size_t s = 0; s < g.masks_causal.size(); ++s)
        set_causal_mask(g.masks_causal[s], segs[s].n, segs[s].position);
    for (ggml_tensor* zero : g.zero_inputs)
        ggml_backend_tensor_memset(zero, 0, 0, ggml_nbytes(zero));
    if (g.ovr_keep) {
        // Local row row0 + t is absolute row position + t of its segment; table column 0 is zero (row keeps its sum).
        const int n_cond = (int)g.m->hp.n_cond;
        std::vector<float> keep(g.n, 1.0f);
        std::vector<int32_t> sel(g.n, 0);
        size_t spk = 0, bkt = 0;
        for (size_t s = 0; s < segs.size(); ++s) {
            const auto& sg = segs[s];
            const auto& c = *sg.cond;
            for (int t = 0; t < sg.n; ++t) {
                const int row = sg.position + t;
                if (g.ovr_spk_col[s] >= 0 && row == c.speaker_row) {
                    keep[sg.row0 + t] = 0.0f;
                    sel[sg.row0 + t] = g.ovr_spk_col[s];
                } else if (g.ovr_cond_col[s] >= 0 && row >= c.condition_row && row < c.condition_row + n_cond) {
                    keep[sg.row0 + t] = 0.0f;
                    sel[sg.row0 + t] = g.ovr_cond_col[s] + (row - c.condition_row);
                }
            }
            if (g.ovr_spk_col[s] >= 0)
                ggml_backend_tensor_set(g.ovr_spks[spk++], c.speaker.data(), 0, c.speaker.size() * sizeof(float));
            if (g.ovr_cond_col[s] >= 0)
                ggml_backend_tensor_set(g.ovr_bkts[bkt++], c.buckets.data(), 0, c.buckets.size() * sizeof(int32_t));
        }
        ggml_backend_tensor_set(g.ovr_keep, keep.data(), 0, keep.size() * sizeof(float));
        ggml_backend_tensor_set(g.ovr_sel, sel.data(), 0, sel.size() * sizeof(int32_t));
    }
    if (!g.decode && g.io && g.io->accent) {
        std::vector<std::pair<int, float>> accent(g.n, {-1, 0.0f});
        for (const auto& s : segs)
            if (s.cond)
                std::fill_n(accent.begin() + s.row0, s.n,
                            std::pair<int, float>{s.cond->accent, s.cond->accent_strength});
        set_accent_inputs(*g.m, *g.io, g.n, accent);
    }
}

// The optional conditioning feature (accent adapter).
static bool features_ok(const zonos3_model& m, const zonos3_conditioning& c) {
    if (c.accent >= 0 && (!m.accents || c.accent >= m.accents->n || !std::isfinite(c.accent_strength))) {
        fprintf(stderr, "zonos3: accent adapter %d is not loaded (or its strength is not finite)\n", c.accent);
        return false;
    }
    return true;
}

// Shape/range sanity of the conditioning that does not need the full id sequence.
static bool cond_shape_ok(const zonos3_model& m, const zonos3_conditioning& c) {
    const auto& hp = m.hp;
    if (c.speaker_row >= 0 && c.speaker.size() != hp.speaker_dim) {
        fprintf(stderr, "zonos3: expected a %u-value speaker embedding\n", hp.speaker_dim);
        return false;
    }
    if (c.condition_row >= 0) {
        if (c.buckets.size() != hp.n_cond) {
            fprintf(stderr, "zonos3: expected %u condition buckets\n", hp.n_cond);
            return false;
        }
        for (uint32_t i = 0; i < hp.n_cond; ++i)
            if (c.buckets[i] < 0 || c.buckets[i] > (int32_t)hp.cond_bucket_counts[i]) {
                fprintf(stderr, "zonos3: condition bucket %d for %s outside 0..%u (%u = unknown)\n", c.buckets[i],
                        hp.cond_names[i].c_str(), hp.cond_bucket_counts[i] - 1, hp.cond_bucket_counts[i]);
                return false;
            }
    }
    return features_ok(m, c);
}

static ggml_tensor* prefill_run(const zonos3_model& m, const float* ids, int n_tokens, const zonos3_conditioning& cond,
                                bool capture, int n_layer_limit, gctx& g, ggml_context*& ctx, ggml_gallocr_t& galloc) {
    if (!valid_ids(m, ids, n_tokens) || !zonos3_check_conditioning(m, ids, n_tokens, cond) || !features_ok(m, cond))
        return nullptr;
    ctx = ggml_init({(size_t)128 * 1024 * 1024, nullptr, true});
    if (!ctx)
        return nullptr;
    g.m = &m;
    g.ctx = ctx;
    g.n = n_tokens;
    g.capture = capture;
    g.cond = &cond;
    g.gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_tensor* output = build_graph(g, n_layer_limit);
    g.finish(output);
    galloc = ggml_gallocr_new(m.buft);
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, g.gf)) {
        fprintf(stderr, "zonos3: prefill allocation failed\n");
        return nullptr;
    }
    set_graph_inputs(g, ids);
    if (ggml_backend_graph_compute(m.backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "zonos3: prefill computation failed\n");
        return nullptr;
    }
    return output;
}

} // namespace

bool zonos3_validate(const zonos3_model& m, const float* ids, int n_tokens, const zonos3_conditioning& cond,
                     const char* out_dir, int n_layer_limit) {
    gctx g;
    ggml_context* ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    bool ok = prefill_run(m, ids, n_tokens, cond, true, n_layer_limit, g, ctx, galloc) != nullptr;
    if (ok) {
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        ok = !ec;
        for (const auto& c : g.caps)
            if (ok)
                ok = save_tensor(out_dir, c.first, c.second);
        if (ok)
            fprintf(stderr, "zonos3: validation wrote %zu tensors to %s\n", g.caps.size(), out_dir);
    }
    if (galloc)
        ggml_gallocr_free(galloc);
    if (ctx)
        ggml_free(ctx);
    return ok;
}

bool zonos3_logits(const zonos3_model& m, const float* ids, int n_tokens, const zonos3_conditioning& cond,
                   std::vector<float>& out_logits) {
    gctx g;
    ggml_context* ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor* logits = prefill_run(m, ids, n_tokens, cond, false, -1, g, ctx, galloc);
    if (logits) {
        out_logits.resize((size_t)m.hp.audio_vocab * m.hp.n_codebooks * n_tokens);
        ggml_backend_tensor_get(logits, out_logits.data(), 0, out_logits.size() * sizeof(float));
    }
    if (galloc)
        ggml_gallocr_free(galloc);
    if (ctx)
        ggml_free(ctx);
    return logits != nullptr;
}

bool zonos3_moe_capture(const zonos3_model& m, const float* ids, int n_tokens, const zonos3_conditioning& cond,
                        std::vector<zonos3_moe_act>& out) {
    gctx g;
    g.cap_moe = true;
    ggml_context* ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    bool ok = prefill_run(m, ids, n_tokens, cond, false, -1, g, ctx, galloc) != nullptr;
    if (ok) {
        std::map<int, zonos3_moe_act> layers;
        for (const auto& c : g.caps) {
            const int L = atoi(c.first.c_str() + c.first.rfind('_') + 1);
            ggml_tensor* t = c.second;
            auto& a = layers[L];
            a.layer = L;
            if (c.first.rfind("moe_in_", 0) == 0) {
                a.n_embd = (int)t->ne[0];
                a.n = (int)t->ne[1];
                a.moe_in.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, a.moe_in.data(), 0, ggml_nbytes(t));
            } else if (c.first.rfind("moe_y_", 0) == 0) {
                a.n_ff = (int)t->ne[0];
                a.k = (int)t->ne[1];
                a.n = (int)t->ne[2];
                a.moe_y.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, a.moe_y.data(), 0, ggml_nbytes(t));
            } else if (c.first.rfind("moe_sel_", 0) == 0) {
                a.sel.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, a.sel.data(), 0, ggml_nbytes(t));
            }
        }
        for (auto& layer : layers)
            out.push_back(std::move(layer.second));
    }
    if (galloc)
        ggml_gallocr_free(galloc);
    if (ctx)
        ggml_free(ctx);
    return ok;
}

static int generate_recompute(const zonos3_model& m, const std::vector<float>& prompt_ids, const zonos3_prompt& prompt,
                              const std::vector<float>& twin_ids, const zonos3_prompt* twin, float cfg_scale,
                              int max_frames, const zonos3_sampling& sp, std::vector<int32_t>& out_codes,
                              int& eos_frame, const zonos3_frame_cb& on_frame) {
    const int W = (int)m.hp.n_codebooks + 1, av = (int)m.hp.audio_vocab, ncb = (int)m.hp.n_codebooks;
    const int n0 = prompt.n_rows;
    std::vector<float> sequence(prompt_ids), twin_sequence(twin_ids), logits, twin_logits, guided((size_t)av * ncb);
    sequence.reserve((size_t)(n0 + max_frames) * W);
    if (twin)
        twin_sequence.reserve((size_t)(twin->n_rows + max_frames) * W);
    zonos3_sampler sampler(sp, m.hp);
    zonos3_slot slot;
    slot.active = true;
    slot.prefill = prompt.prefill_frames;
    for (int step = 0; step < max_frames; ++step) {
        const int n = (int)sequence.size() / W;
        if (!zonos3_logits(m, sequence.data(), n, prompt.cond, logits)) {
            eos_frame = -1;
            return -1;
        }
        const float* source = logits.data() + (size_t)(n - 1) * av * ncb;
        if (twin) {
            const int nt = (int)twin_sequence.size() / W;
            if (!zonos3_logits(m, twin_sequence.data(), nt, twin->cond, twin_logits)) {
                eos_frame = -1;
                return -1;
            }
            zonos3_cfg_mix(source, twin_logits.data() + (size_t)(nt - 1) * av * ncb, cfg_scale, guided.size(),
                           guided.data());
            source = guided.data();
        }
        zonos3_slot_sample(m, slot, sampler, source, out_codes, max_frames, on_frame);
        if (slot.done)
            break;
        sequence.insert(sequence.end(), slot.next_ids.begin(), slot.next_ids.end());
        if (twin)
            twin_sequence.insert(twin_sequence.end(), slot.next_ids.begin(), slot.next_ids.end());
    }
    eos_frame = slot.eos_frame;
    return slot.step;
}

static int round_up_256(int x) {
    return ((x + 255) / 256) * 256;
}

// Global (attention) layers that run, in probe order.
static std::vector<int> attention_layers(const zonos3_model& m) {
    std::vector<int> out;
    for (int L = 0; L < (int)m.hp.n_layer; ++L)
        if (!m.layers[L].is_gdn && !(L < (int)m.layer_skip.size() && m.layer_skip[L]))
            out.push_back(L);
    return out;
}

// Decode-graph feature keys: bit 0 = alignment probe, bit 1 = accent adapters, bits 2.. = masked-probe slots.
constexpr int FEATURE_PROBE = 1, FEATURE_ACCENT = 2;

static bool build_decode_graph(zonos3_batch_ctx& bc, const zonos3_model& m, int width, zonos3_batch_ctx::dec_graph& dg,
                               int key = 0) {
    dg.width = width;
    dg.ctx = ggml_init({(size_t)64 * 1024 * 1024, nullptr, true});
    if (!dg.ctx)
        return false;
    gctx g;
    g.m = &m;
    g.ctx = dg.ctx;
    g.n = width;
    g.capture = false;
    g.kc = &bc.k_cache;
    g.vc = &bc.v_cache;
    g.state = &bc.gdn_state;
    g.history = &bc.gdn_history;
    g.decode = true;
    g.n_slots = width;
    g.slot_cap = bc.slot_cap;
    g.steer_sources = bc.steer_sources;
    if (key) {
        g.io = std::make_shared<zonos3_feature_io>();
        g.io->accent = (key & FEATURE_ACCENT) != 0;
        if (key & FEATURE_PROBE) {
            g.io->lmax = bc.features.probe_max_text;
            g.io->masked = key >> 2;
            g.io->attn_layers = attention_layers(m);
        }
    }
    dg.gf = ggml_new_graph_custom(dg.ctx, 32768, false);
    g.gf = dg.gf;
    ggml_tensor* logits = build_graph(g, -1);
    g.finish(logits);
    dg.galloc = ggml_gallocr_new(m.buft);
    if (!dg.galloc || !ggml_gallocr_alloc_graph(dg.galloc, dg.gf))
        return false;
    dg.ids = g.ids;
    dg.pos_rope = g.pos_rope;
    dg.pos_cache = g.pos_cache;
    dg.steer = g.steer;
    dg.mask = g.mask;
    dg.active = g.active;
    dg.logits = logits;
    dg.io = g.io;
    return true;
}

static std::vector<int> ladder_widths(int n_slots) {
    std::vector<int> widths;
    for (int w = 1; w < n_slots; w *= 2)
        widths.push_back(w);
    widths.push_back(n_slots);
    return widths;
}

static void free_ladder(std::vector<zonos3_batch_ctx::dec_graph>& ladder) {
    for (auto& dg : ladder) {
        if (dg.galloc)
            ggml_gallocr_free(dg.galloc);
        if (dg.ctx)
            ggml_free(dg.ctx);
    }
    ladder.clear();
}

bool zonos3_batch_init(zonos3_batch_ctx& bc, const zonos3_model& m, int n_slots, int capacity,
                       const std::vector<int>& steer_sources, const zonos3_batch_features& features) {
    zonos3_batch_free(bc);
    if (n_slots <= 0 || n_slots > 1024 || capacity <= 0 || capacity > (int)m.hp.n_ctx_train) {
        fprintf(stderr, "zonos3: invalid batch capacity\n");
        return false;
    }
    if (!std::is_sorted(steer_sources.begin(), steer_sources.end(), std::less_equal<int>()) ||
        (!steer_sources.empty() && (steer_sources.front() < 0 || steer_sources.back() > 2 * (int)m.hp.n_layer))) {
        fprintf(stderr, "zonos3: steering sources must be ascending Full AttnRes bank indices in 0..%d\n",
                2 * (int)m.hp.n_layer);
        return false;
    }
    const auto& hp = m.hp;
    bc.model = &m;
    bc.n_slots = n_slots;
    bc.slot_cap = round_up_256(capacity);
    bc.steer_sources = steer_sources;
    bc.features = features;
    if (features.probe_max_text < 0 || features.probe_max_text > bc.slot_cap) {
        fprintf(stderr, "zonos3: probe text span limit must be within 0..%d\n", bc.slot_cap);
        zonos3_batch_free(bc);
        return false;
    }
    if (features.probe_max_text > 0 && attention_layers(m).empty()) {
        fprintf(stderr, "zonos3: the alignment probe needs a global attention layer\n");
        zonos3_batch_free(bc);
        return false;
    }
    bc.h_steer.assign(steer_sources.size() * (size_t)n_slots * hp.n_embd, 0.0f);
    bc.W = (int)hp.n_codebooks + 1;
    bc.ncb = (int)hp.n_codebooks;
    bc.av = (int)hp.audio_vocab;
    const int nl = (int)hp.n_layer, hd = zonos3_kv_head_dim(hp);
    const int H = (int)hp.gdn_heads, S = (int)hp.gdn_head_dim, I = H * S, R = (int)hp.gdn_conv_kernel - 1;
    bc.ctx_kv = ggml_init({ggml_tensor_overhead() * (size_t)(2 * nl + 8), nullptr, true});
    if (!bc.ctx_kv) {
        zonos3_batch_free(bc);
        return false;
    }
    bc.k_cache.resize(nl);
    bc.v_cache.resize(nl);
    bc.gdn_state.resize(nl);
    bc.gdn_history.resize(nl);
    // Determine the full-precision diagnostic profile from the actual complete
    // weight inventory, not a filename or a single unquantized projection.
    const bool full_f32 =
        !m.tensors.empty() &&
        std::all_of(m.tensors.begin(), m.tensors.end(), [](const std::pair<const std::string, ggml_tensor*>& weight) {
            return weight.second->type == GGML_TYPE_F32;
        });
    const ggml_type kv_type = m.is_gpu && !full_f32 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    for (int L = 0; L < nl; ++L) {
        if (m.layers[L].is_gdn) {
            bc.gdn_state[L] = ggml_new_tensor_4d(bc.ctx_kv, GGML_TYPE_F32, S, S, H, n_slots);
            bc.gdn_history[L] = ggml_new_tensor_3d(bc.ctx_kv, GGML_TYPE_F32, R, 3 * I, n_slots);
        } else {
            const int64_t rows = (int64_t)n_slots * (bc.slot_cap + 1);
            bc.k_cache[L] = ggml_new_tensor_3d(bc.ctx_kv, kv_type, hd, rows, hp.n_head_kv);
            bc.v_cache[L] = ggml_new_tensor_3d(bc.ctx_kv, kv_type, hd, rows, hp.n_head_kv);
        }
    }
    bc.buf_kv = ggml_backend_alloc_ctx_tensors(bc.ctx_kv, m.backend);
    if (!bc.buf_kv) {
        zonos3_batch_free(bc);
        return false;
    }
    ggml_backend_buffer_clear(bc.buf_kv, 0);
    const std::vector<int> widths = ladder_widths(n_slots);
    bc.dec_ladder.resize(widths.size());
    for (size_t i = 0; i < widths.size(); ++i)
        if (!build_decode_graph(bc, m, widths[i], bc.dec_ladder[i])) {
            zonos3_batch_free(bc);
            return false;
        }
    bc.galloc_prefill = ggml_gallocr_new(m.buft);
    bc.h_ids.resize((size_t)bc.W * n_slots);
    bc.h_pos_rope.resize(n_slots);
    bc.h_pos_cache.resize(n_slots);
    bc.h_mask.resize((size_t)bc.slot_cap * n_slots);
    bc.h_active.resize(n_slots);
    bc.h_logits.resize((size_t)bc.av * bc.ncb * n_slots);
    if (!bc.galloc_prefill) {
        zonos3_batch_free(bc);
        return false;
    }
    return true;
}

void zonos3_batch_free(zonos3_batch_ctx& bc) {
    free_ladder(bc.dec_ladder);
    for (auto& variant : bc.dec_variants)
        free_ladder(variant.ladder);
    if (bc.galloc_prefill)
        ggml_gallocr_free(bc.galloc_prefill);
    if (bc.buf_kv)
        ggml_backend_buffer_free(bc.buf_kv);
    if (bc.ctx_kv)
        ggml_free(bc.ctx_kv);
    bc = zonos3_batch_ctx();
}

static void reset_slot(zonos3_batch_ctx& bc, int slot) {
    const auto& hp = bc.model->hp;
    for (int L = 0; L < (int)hp.n_layer; ++L) {
        if (bc.gdn_state[L]) {
            auto *state = bc.gdn_state[L], *history = bc.gdn_history[L];
            ggml_backend_tensor_memset(state, 0, (size_t)slot * state->nb[3], state->nb[3]);
            ggml_backend_tensor_memset(history, 0, (size_t)slot * history->nb[2], history->nb[2]);
        } else {
            for (ggml_tensor* cache : {bc.k_cache[L], bc.v_cache[L]}) {
                const size_t bytes = (size_t)(bc.slot_cap + 1) * cache->nb[1];
                for (int h = 0; h < (int)hp.n_head_kv; ++h)
                    ggml_backend_tensor_memset(cache, 0, (size_t)h * cache->nb[2] + (size_t)slot * bytes, bytes);
            }
        }
    }
}

bool zonos3_batch_slot_extend(zonos3_batch_ctx& bc, int slot, const float* ids, int n, int position,
                              const zonos3_conditioning& cond, float* out_logits) {
    if (!bc.model || slot < 0 || slot >= bc.n_slots || position < 0 || n <= 0 || position > bc.slot_cap - n ||
        position + n > (int)bc.model->hp.n_ctx_train || !out_logits || !valid_ids(*bc.model, ids, n) ||
        !cond_shape_ok(*bc.model, cond))
        return false;
    const auto& m = *bc.model;
    ggml_context* ctx = ggml_init({(size_t)128 * 1024 * 1024, nullptr, true});
    if (!ctx)
        return false;
    gctx g;
    g.m = &m;
    g.ctx = ctx;
    g.n = n;
    g.capture = false;
    g.kc = &bc.k_cache;
    g.vc = &bc.v_cache;
    g.state = &bc.gdn_state;
    g.history = &bc.gdn_history;
    g.slot_cap = bc.slot_cap;
    g.cache_slot = slot;
    g.position = position;
    g.cond = &cond;
    g.gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_tensor* logits = build_graph(g, -1);
    g.finish(logits);
    bool ok = ggml_gallocr_alloc_graph(bc.galloc_prefill, g.gf);
    if (ok) {
        set_graph_inputs(g, ids);
        ok = ggml_backend_graph_compute(m.backend, g.gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            const size_t bytes = (size_t)bc.av * bc.ncb * sizeof(float);
            ggml_backend_tensor_get(logits, out_logits, (size_t)(n - 1) * bytes, bytes);
            memcpy(bc.h_logits.data() + (size_t)slot * bc.av * bc.ncb, out_logits, bytes);
        }
    }
    ggml_free(ctx);
    if (!ok)
        fprintf(stderr, "zonos3: slot prefill failed\n");
    return ok;
}

bool zonos3_batch_slot_prefill(zonos3_batch_ctx& bc, int slot, const float* ids, int n, const zonos3_conditioning& cond,
                               float* out_logits) {
    if (!bc.model || slot < 0 || slot >= bc.n_slots || n <= 0 || n > bc.slot_cap || n > (int)bc.model->hp.n_ctx_train ||
        !valid_ids(*bc.model, ids, n) || !zonos3_check_conditioning(*bc.model, ids, n, cond))
        return false;
    reset_slot(bc, slot);
    constexpr int chunk = 256;
    for (int position = 0; position < n; position += chunk) {
        const int count = std::min(chunk, n - position);
        if (!zonos3_batch_slot_extend(bc, slot, ids + (size_t)position * bc.W, count, position, cond, out_logits))
            return false;
    }
    return true;
}

bool zonos3_batch_pair_prefill(zonos3_batch_ctx& bc, int slot_a, const float* ids_a, int n_a,
                               const zonos3_conditioning& cond_a, float* out_logits_a, int slot_b, const float* ids_b,
                               int n_b, const zonos3_conditioning& cond_b, float* out_logits_b) {
    if (!bc.model || slot_a == slot_b || !out_logits_a || !out_logits_b)
        return false;
    const auto& m = *bc.model;
    struct seq {
        int slot;
        const float* ids;
        int n;
        const zonos3_conditioning* cond;
        float* out;
    };
    const seq seqs[2] = {{slot_a, ids_a, n_a, &cond_a, out_logits_a}, {slot_b, ids_b, n_b, &cond_b, out_logits_b}};
    for (const seq& s : seqs)
        if (s.slot < 0 || s.slot >= bc.n_slots || s.n <= 0 || s.n > bc.slot_cap || s.n > (int)m.hp.n_ctx_train ||
            !valid_ids(m, s.ids, s.n) || !zonos3_check_conditioning(m, s.ids, s.n, *s.cond) || !features_ok(m, *s.cond))
            return false;
    reset_slot(bc, slot_a);
    reset_slot(bc, slot_b);
    constexpr int chunk = 256;
    const size_t bytes = (size_t)bc.av * bc.ncb * sizeof(float);
    std::vector<float> rows;
    for (int position = 0; position < std::max(n_a, n_b); position += chunk) {
        std::vector<gctx::segment> segs;
        std::vector<const seq*> owners;
        int total = 0;
        for (const seq& s : seqs) {
            const int count = std::clamp(s.n - position, 0, chunk);
            if (count == 0)
                continue; // this prompt finished in an earlier chunk
            segs.push_back({total, count, s.slot, position, s.cond});
            owners.push_back(&s);
            total += count;
        }
        if (segs.size() == 1) {
            const seq& s = *owners[0];
            if (!zonos3_batch_slot_extend(bc, s.slot, s.ids + (size_t)position * bc.W, segs[0].n, position, *s.cond,
                                          s.out))
                return false;
            continue;
        }
        rows.resize((size_t)total * bc.W);
        for (size_t i = 0; i < segs.size(); ++i)
            std::copy_n(owners[i]->ids + (size_t)position * bc.W, (size_t)segs[i].n * bc.W,
                        rows.begin() + (size_t)segs[i].row0 * bc.W);
        ggml_context* ctx = ggml_init({(size_t)128 * 1024 * 1024, nullptr, true});
        if (!ctx)
            return false;
        gctx g;
        g.m = &m;
        g.ctx = ctx;
        g.n = total;
        g.capture = false;
        g.kc = &bc.k_cache;
        g.vc = &bc.v_cache;
        g.state = &bc.gdn_state;
        g.history = &bc.gdn_history;
        g.slot_cap = bc.slot_cap;
        g.segs = segs;
        g.gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_tensor* logits = build_graph(g, -1);
        g.finish(logits);
        bool ok = ggml_gallocr_alloc_graph(bc.galloc_prefill, g.gf);
        if (ok) {
            set_graph_inputs(g, rows.data());
            ok = ggml_backend_graph_compute(m.backend, g.gf) == GGML_STATUS_SUCCESS;
        }
        for (size_t i = 0; ok && i < segs.size(); ++i) {
            const seq& s = *owners[i];
            if (position + segs[i].n != s.n)
                continue; // the prompt continues in the next chunk
            ggml_backend_tensor_get(logits, s.out, (size_t)(segs[i].row0 + segs[i].n - 1) * bytes, bytes);
            memcpy(bc.h_logits.data() + (size_t)s.slot * bc.av * bc.ncb, s.out, bytes);
        }
        ggml_free(ctx);
        if (!ok) {
            fprintf(stderr, "zonos3: pair prefill failed\n");
            return false;
        }
    }
    return true;
}

// clip_window (zonos3.tts.probe): an inclusive text window inside [0, text_len).
static std::pair<int, int> clip_window(int lo, int hi, int text_len) {
    lo = std::max(0, std::min(lo, text_len - 1));
    return {lo, std::min(text_len - 1, std::max(hi, lo))};
}

static bool probe_ok(const zonos3_batch_ctx& bc, const zonos3_slot& s, const std::vector<int>& layers) {
    const auto& p = s.probe;
    const int nh = (int)bc.model->hp.n_head;
    if (p.text_len > bc.features.probe_max_text || p.text_start < 0 || p.text_start + p.text_len > s.n_past) {
        fprintf(stderr,
                "zonos3: alignment probe text span [%d, +%d) must lie in the slot's %d cached rows and "
                "hold at most %d rows (zonos3_batch_features::probe_max_text)\n",
                p.text_start, p.text_len, s.n_past, bc.features.probe_max_text);
        return false;
    }
    auto head_ok = [&](int layer, int head) {
        return std::find(layers.begin(), layers.end(), layer) != layers.end() && head >= 0 && head < nh;
    };
    for (const auto& h : p.heads)
        if (!head_ok(h.layer, h.head)) {
            fprintf(stderr, "zonos3: align probe head (%d, %d) is not an attention head of this model\n", h.layer,
                    h.head);
            return false;
        }
    for (const auto& w : p.windows)
        if (!head_ok(w.layer, w.head)) {
            fprintf(stderr, "zonos3: align probe window head (%d, %d) is not an attention head of this model\n",
                    w.layer, w.head);
            return false;
        }
    return true;
}

// Probe and accent inputs of a feature decode graph. `cols[c]` is the live slot of column c (or null).
static void set_feature_inputs(const zonos3_batch_ctx& bc, const zonos3_feature_io& io, int width,
                               const std::vector<const zonos3_slot*>& cols) {
    const auto& m = *bc.model;
    if (io.accent) {
        std::vector<std::pair<int, float>> sel(width, {-1, 0.0f});
        for (int c = 0; c < width; ++c)
            if (cols[c] && accent_on(m, cols[c]->accent, cols[c]->accent_strength))
                sel[c] = {cols[c]->accent, cols[c]->accent_strength};
        set_accent_inputs(m, io, width, sel);
    }
    if (!io.lmax)
        return;
    const int lmax = io.lmax, nh = (int)m.hp.n_head, nkv = (int)m.hp.n_head_kv, S = bc.slot_cap;
    const size_t nl = io.attn_layers.size();
    auto layer_slot = [&](int layer) {
        return (size_t)(std::find(io.attn_layers.begin(), io.attn_layers.end(), layer) - io.attn_layers.begin());
    };
    auto probing = [&](const zonos3_slot* s) { return s && s->probe.enabled(); };
    std::vector<int32_t> kidx((size_t)lmax * width * nkv);
    std::vector<float> tpos(lmax), w(nl * width * nh, 0.0f), lo(nl * width * nh, 0.0f), hi(nl * width * nh, 0.0f);
    for (int j = 0; j < lmax; ++j)
        tpos[j] = (float)j;
    std::vector<int> masked;
    for (int c = 0; c < width; ++c) {
        const zonos3_slot* s = probing(cols[c]) ? cols[c] : nullptr;
        const int ts = s ? s->probe.text_start : 0, len = s ? s->probe.text_len : 1;
        for (int h = 0; h < nkv; ++h)
            for (int j = 0; j < lmax; ++j)
                kidx[((size_t)h * width + c) * lmax + j] = c * (S + 1) + std::min(ts + j, S);
        for (size_t l = 0; l < nl; ++l)
            std::fill_n(hi.begin() + (l * width + c) * nh, nh, (float)(len - 1));
        if (!s)
            continue;
        for (const auto& head : s->probe.heads)
            w[(layer_slot(head.layer) * width + c) * nh + head.head] = 1.0f;
        for (const auto& win : s->probe.windows) {
            const auto clipped = clip_window(win.lo, win.hi, len);
            const size_t at = (layer_slot(win.layer) * width + c) * nh + win.head;
            lo[at] = (float)clipped.first;
            hi[at] = (float)clipped.second;
        }
        if (!s->probe.windows.empty())
            masked.push_back(c);
    }
    ggml_backend_tensor_set(io.pr_kidx, kidx.data(), 0, kidx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(io.pr_tpos, tpos.data(), 0, tpos.size() * sizeof(float));
    for (size_t l = 0; l < nl; ++l) {
        const size_t off = l * width * nh, bytes = (size_t)width * nh * sizeof(float);
        ggml_backend_tensor_set(io.pr_w[l], w.data() + off, 0, bytes);
        ggml_backend_tensor_set(io.pr_lo[l], lo.data() + off, 0, bytes);
        ggml_backend_tensor_set(io.pr_hi[l], hi.data() + off, 0, bytes);
    }
    if (!io.masked)
        return;
    // Unused masked rows repeat the first masked slot: identical inputs, identical writes.
    const int M = io.masked;
    while ((int)masked.size() < M)
        masked.push_back(masked.front());
    std::vector<int32_t> rows(M), mkidx((size_t)M * S * nkv);
    std::vector<float> kpos(S), npast(M), mts(M), mtl(M), flag(nl * M * nh, 0.0f), mlo(nl * M * nh, 0.0f),
        mhi(nl * M * nh, 0.0f);
    for (int j = 0; j < S; ++j)
        kpos[j] = (float)j;
    for (int i = 0; i < M; ++i) {
        const int c = masked[i];
        const zonos3_slot& s = *cols[c];
        const int len = s.probe.text_len;
        rows[i] = c;
        for (int h = 0; h < nkv; ++h)
            for (int j = 0; j < S; ++j)
                mkidx[((size_t)h * M + i) * S + j] = c * (S + 1) + j;
        npast[i] = (float)s.n_past;
        mts[i] = (float)s.probe.text_start;
        mtl[i] = (float)len;
        for (size_t l = 0; l < nl; ++l)
            std::fill_n(mhi.begin() + (l * M + i) * nh, nh, (float)(len - 1));
        for (const auto& win : s.probe.windows) {
            const auto clipped = clip_window(win.lo, win.hi, len);
            const size_t at = (layer_slot(win.layer) * M + i) * nh + win.head;
            flag[at] = 1.0f;
            mlo[at] = (float)clipped.first;
            mhi[at] = (float)clipped.second;
        }
    }
    ggml_backend_tensor_set(io.pm_rows, rows.data(), 0, rows.size() * sizeof(int32_t));
    ggml_backend_tensor_set(io.pm_kidx, mkidx.data(), 0, mkidx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(io.pm_kpos, kpos.data(), 0, kpos.size() * sizeof(float));
    ggml_backend_tensor_set(io.pm_npast, npast.data(), 0, npast.size() * sizeof(float));
    ggml_backend_tensor_set(io.pm_ts, mts.data(), 0, mts.size() * sizeof(float));
    ggml_backend_tensor_set(io.pm_tl, mtl.data(), 0, mtl.size() * sizeof(float));
    for (size_t l = 0; l < nl; ++l) {
        const size_t off = l * M * nh, bytes = (size_t)M * nh * sizeof(float);
        ggml_backend_tensor_set(io.pm_flag[l], flag.data() + off, 0, bytes);
        ggml_backend_tensor_set(io.pm_lo[l], mlo.data() + off, 0, bytes);
        ggml_backend_tensor_set(io.pm_hi[l], mhi.data() + off, 0, bytes);
    }
}

bool zonos3_batch_step(zonos3_batch_ctx& bc, const std::vector<zonos3_slot*>& active) {
    const auto& m = *bc.model;
    int extent = 0;
    for (auto* slot : active)
        if (slot && slot->active && !slot->done) {
            if (slot->index < 0 || slot->index >= bc.n_slots || slot->n_past < 0 || slot->n_past >= bc.slot_cap ||
                slot->n_past >= (int)m.hp.n_ctx_train || slot->next_ids.size() != (size_t)bc.W) {
                fprintf(stderr, "zonos3: invalid active decode slot\n");
                return false;
            }
            extent = std::max(extent, slot->index + 1);
        }
    if (!extent)
        return true;
    const size_t K = bc.steer_sources.size(), D = m.hp.n_embd;
    for (auto* slot : active)
        if (slot && slot->active && !slot->done && !slot->steer.empty() && slot->steer.size() != K * D) {
            fprintf(stderr, "zonos3: slot steering delta must hold %d sources x %d values\n", (int)K, (int)D);
            return false;
        }
    // Optional features: alignment probe (masked heads), accent adapters.
    const std::vector<int> attn_layers = attention_layers(m);
    int n_probe = 0, n_masked = 0, n_accent = 0;
    std::vector<const zonos3_slot*> cols(bc.n_slots, nullptr);
    for (auto* slot : active)
        if (slot && slot->active && !slot->done) {
            cols[slot->index] = slot;
            if (slot->probe.enabled()) {
                if (!bc.features.probe_max_text) {
                    fprintf(stderr,
                            "zonos3: alignment probe on a batch without probe support (zonos3_batch_features)\n");
                    return false;
                }
                if (!probe_ok(bc, *slot, attn_layers))
                    return false;
                ++n_probe;
                if (!slot->probe.windows.empty())
                    ++n_masked;
            }
            if (slot->accent >= 0) {
                if (!m.accents || slot->accent >= m.accents->n || !std::isfinite(slot->accent_strength)) {
                    fprintf(stderr, "zonos3: accent adapter %d is not loaded (or its strength is not finite)\n",
                            slot->accent);
                    return false;
                }
                if (accent_on(m, slot->accent, slot->accent_strength))
                    ++n_accent;
            }
        }
    int masked_rows = 0;
    if (n_masked)
        for (masked_rows = 1; masked_rows < n_masked; masked_rows *= 2) {
        }
    const int key = (n_probe ? FEATURE_PROBE : 0) | (n_accent ? FEATURE_ACCENT : 0) | (masked_rows << 2);
    const std::vector<zonos3_batch_ctx::dec_graph>* ladder = &bc.dec_ladder;
    if (key) {
        auto variant = std::find_if(bc.dec_variants.begin(), bc.dec_variants.end(),
                                    [&](const zonos3_batch_ctx::dec_variant& v) { return v.key == key; });
        if (variant == bc.dec_variants.end()) {
            zonos3_batch_ctx::dec_variant built;
            built.key = key;
            const std::vector<int> widths = ladder_widths(bc.n_slots);
            built.ladder.resize(widths.size());
            for (size_t i = 0; i < widths.size(); ++i)
                if (!build_decode_graph(bc, m, widths[i], built.ladder[i], key)) {
                    free_ladder(built.ladder);
                    fprintf(stderr, "zonos3: feature decode graph allocation failed\n");
                    return false;
                }
            bc.dec_variants.push_back(std::move(built));
            variant = bc.dec_variants.end() - 1;
        }
        ladder = &variant->ladder;
    }
    const auto* dg = &ladder->back();
    for (const auto& candidate : *ladder)
        if (candidate.width >= extent) {
            dg = &candidate;
            break;
        }
    const int width = dg->width, N = bc.n_slots, S = bc.slot_cap;
    const uint16_t zero = ggml_fp32_to_fp16(0), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int c = 0; c < width; ++c) {
        for (int k = 0; k < bc.W; ++k)
            bc.h_ids[(size_t)k * N + c] = (int32_t)(k < bc.ncb ? m.hp.audio_pad_id : m.hp.text_vocab);
        bc.h_active[c] = 0;
        bc.h_pos_rope[c] = 0;
        bc.h_pos_cache[c] = c * (S + 1) + S; // Private idle-write row.
        uint16_t* mask = bc.h_mask.data() + (size_t)c * S;
        std::fill(mask, mask + S, ninf);
        mask[0] = zero;
    }
    for (auto* slot : active)
        if (slot && slot->active && !slot->done) {
            const int c = slot->index;
            for (int k = 0; k < bc.W; ++k)
                bc.h_ids[(size_t)k * N + c] = slot->next_ids[k];
            bc.h_active[c] = 1;
            bc.h_pos_rope[c] = slot->n_past;
            bc.h_pos_cache[c] = c * (S + 1) + slot->n_past;
            uint16_t* mask = bc.h_mask.data() + (size_t)c * S;
            std::fill(mask, mask + slot->n_past + 1, zero);
        }
    for (int k = 0; k < bc.W; ++k)
        ggml_backend_tensor_set(dg->ids[k], bc.h_ids.data() + (size_t)k * N, 0, (size_t)width * sizeof(int32_t));
    if (dg->pos_rope)
        ggml_backend_tensor_set(dg->pos_rope, bc.h_pos_rope.data(), 0, (size_t)width * sizeof(int32_t));
    if (dg->pos_cache)
        ggml_backend_tensor_set(dg->pos_cache, bc.h_pos_cache.data(), 0, (size_t)width * sizeof(int32_t));
    ggml_backend_tensor_set(dg->mask, bc.h_mask.data(), 0, (size_t)S * width * sizeof(uint16_t));
    ggml_backend_tensor_set(dg->active, bc.h_active.data(), 0, (size_t)width * sizeof(float));
    if (K) {
        std::fill(bc.h_steer.begin(), bc.h_steer.end(), 0.0f);
        for (auto* slot : active)
            if (slot && slot->active && !slot->done && !slot->steer.empty())
                for (size_t j = 0; j < K; ++j)
                    memcpy(bc.h_steer.data() + (j * N + slot->index) * D, slot->steer.data() + j * D,
                           D * sizeof(float));
        for (size_t j = 0; j < K; ++j)
            ggml_backend_tensor_set(dg->steer[j], bc.h_steer.data() + j * N * D, 0, (size_t)width * D * sizeof(float));
    }
    if (dg->io)
        set_feature_inputs(bc, *dg->io, width, cols);
    if (ggml_backend_graph_compute(m.backend, dg->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "zonos3: decode computation failed\n");
        return false;
    }
    if (dg->io && dg->io->lmax) {
        const int lmax = dg->io->lmax;
        std::vector<float> dist(lmax);
        for (auto* slot : active)
            if (slot && slot->active && !slot->done && slot->probe.enabled()) {
                ggml_backend_tensor_get(dg->io->pr_acc, dist.data(), (size_t)slot->index * lmax * sizeof(float),
                                        (size_t)lmax * sizeof(float));
                slot->readout = zonos3_probe_reduce(dist.data(), slot->probe.text_len);
            }
    }
    // Inactive consumers retain their last logits as well as their model state.
    // Reading all graph columns would expose meaningless idle-query outputs.
    for (auto* slot : active)
        if (slot && slot->active && !slot->done) {
            const size_t bytes = (size_t)bc.av * bc.ncb * sizeof(float);
            const size_t offset = (size_t)slot->index * bytes;
            ggml_backend_tensor_get(dg->logits, bc.h_logits.data() + (size_t)slot->index * bc.av * bc.ncb, offset,
                                    bytes);
        }
    for (auto* slot : active)
        if (slot && slot->active && !slot->done)
            ++slot->n_past;
    return true;
}

const float* zonos3_batch_slot_logits(const zonos3_batch_ctx& bc, int index) {
    return bc.h_logits.data() + (size_t)index * bc.av * bc.ncb;
}

bool zonos3_slot_sample(const zonos3_model& m, zonos3_slot& s, zonos3_sampler& smp, const float* logits,
                        std::vector<int32_t>& out_codes, int max_frames, const zonos3_frame_cb& on_frame) {
    const int ncb = (int)m.hp.n_codebooks, av = (int)m.hp.audio_vocab, W = ncb + 1;
    std::vector<int> frame(ncb);
    for (int cb = 0; cb < ncb; ++cb)
        frame[cb] = smp.sample(&logits[(size_t)cb * av], cb);
    // cold start: codebooks whose delayed stream has not started are forced to audio pad. r is the
    // row index in the audio stream, which already holds `prefill` silence rows.
    const int r = s.prefill + s.step;
    for (int cb = r + 1; cb < ncb; ++cb)
        frame[cb] = (int)m.hp.audio_pad_id;
    smp.accept(frame);
    for (int cb = 0; cb < ncb; ++cb)
        out_codes.push_back(frame[cb]);

    const int frame_idx = s.step;
    bool keep = true;
    if (on_frame) {
        const std::vector<int32_t> fc(frame.begin(), frame.end());
        keep = on_frame(frame_idx, fc.data(), ncb);
    }

    int max_eoa = -1; // the request's stop id: EOA, or span-end for edit infill
    for (int cb = 0; cb < ncb; ++cb)
        if (frame[cb] == smp.stop_id)
            max_eoa = cb;
    if (s.eos_frame < 0 && max_eoa >= 0) {
        s.eos_frame = std::max(0, r - max_eoa);
        s.countdown = ncb + 1;
    }
    const bool finished = (s.countdown > 0 && --s.countdown == 0);

    s.next_ids.resize(W);
    for (int cb = 0; cb < ncb; ++cb)
        s.next_ids[cb] = frame[cb];
    s.next_ids[ncb] = (int32_t)m.hp.text_vocab;
    s.step += 1;
    if (!keep || finished || s.step >= max_frames)
        s.done = true;
    return keep;
}

void zonos3_cfg_mix(const float* cond, const float* uncond, float scale, size_t n, float* out) {
    for (size_t i = 0; i < n; ++i)
        out[i] = uncond[i] + scale * (cond[i] - uncond[i]);
}

// CLI and server share the same per-slot state/cache implementation. With a CFG twin, slot 0 is the
// conditional prompt and slot 1 the twin; both always receive slot 0's sampled frame.
static int generate_kv(const zonos3_model& m, const std::vector<float>& prompt_ids, const zonos3_prompt& prompt,
                       const std::vector<float>& twin_ids, const zonos3_prompt* twin, float cfg_scale, int max_frames,
                       const zonos3_sampling& sp, std::vector<int32_t>& out_codes, int& eos_frame,
                       const zonos3_frame_cb& on_frame, const zonos3_steering* steering,
                       const zonos3_generate_features* features) {
    const int ncb = (int)m.hp.n_codebooks, av = (int)m.hp.audio_vocab;
    const int n0 = prompt.n_rows, n_twin = twin ? twin->n_rows : 0;
    zonos3_batch_ctx bc;
    const bool steered = steering && steering->hidden_active();
    const bool probed = features && features->probe.enabled();
    zonos3_batch_features batch_features;
    if (probed)
        batch_features.probe_max_text = features->probe.text_len;
    if (!zonos3_batch_init(bc, m, twin ? 2 : 1, std::max(n0, n_twin) + max_frames,
                           steered ? steering->sources : std::vector<int>{}, batch_features))
        return -1;
    zonos3_sampler sampler(sp, m.hp);
    zonos3_slot slot, slot_twin;
    slot.index = 0;
    slot.active = true;
    slot.n_past = n0;
    slot.prefill = prompt.prefill_frames;
    slot.accent = prompt.cond.accent;
    slot.accent_strength = prompt.cond.accent_strength;
    if (probed)
        slot.probe = features->probe;
    std::vector<float> logits((size_t)av * ncb), twin_logits, guided;
    std::vector<zonos3_slot*> active{&slot};
    if (twin) {
        slot_twin.index = 1;
        slot_twin.active = true;
        slot_twin.n_past = n_twin;
        slot_twin.prefill = prompt.prefill_frames;
        slot_twin.accent = twin->cond.accent;
        slot_twin.accent_strength = twin->cond.accent_strength;
        twin_logits.resize(logits.size());
        guided.resize(logits.size());
        if (!zonos3_batch_pair_prefill(bc, 0, prompt_ids.data(), n0, prompt.cond, logits.data(), 1, twin_ids.data(),
                                       n_twin, twin->cond, twin_logits.data())) {
            zonos3_batch_free(bc);
            return -1;
        }
        active.push_back(&slot_twin);
    } else if (!zonos3_batch_slot_prefill(bc, 0, prompt_ids.data(), n0, prompt.cond, logits.data())) {
        zonos3_batch_free(bc);
        return -1;
    }
    if (features && features->step_logits)
        features->step_logits->insert(features->step_logits->end(), logits.begin(), logits.end());
    int steps = 0;
    const auto start = std::chrono::steady_clock::now();
    while (!slot.done) {
        const float* source = logits.data();
        if (twin) {
            zonos3_cfg_mix(logits.data(), twin_logits.data(), cfg_scale, logits.size(), guided.data());
            source = guided.data();
        }
        zonos3_slot_sample(m, slot, sampler, source, out_codes, max_frames, on_frame);
        if (slot.done)
            break;
        if (twin) {
            slot_twin.next_ids = slot.next_ids;
            slot_twin.done = slot.done;
        }
        if (steered) {
            steering->delta_at(slot.step, slot.steer);
            if (twin)
                slot_twin.steer = slot.steer;
        }
        if (!zonos3_batch_step(bc, active)) {
            zonos3_batch_free(bc);
            return -1;
        }
        memcpy(logits.data(), zonos3_batch_slot_logits(bc, 0), logits.size() * sizeof(float));
        if (features && features->step_logits)
            features->step_logits->insert(features->step_logits->end(), logits.begin(), logits.end());
        if (probed && features->readouts)
            features->readouts->push_back(slot.readout);
        if (twin)
            memcpy(twin_logits.data(), zonos3_batch_slot_logits(bc, 1), twin_logits.size() * sizeof(float));
        ++steps;
    }
    eos_frame = slot.eos_frame;
    if (steps) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        fprintf(stderr, "generate(kv%s): %d decode steps in %.3fs = %.1f frames/s, RTF=%.3f\n", twin ? "+cfg" : "",
                steps, elapsed, steps / elapsed, elapsed * ZONOS3_FRAME_RATE / steps);
    }
    const int count = slot.step;
    zonos3_batch_free(bc);
    return count;
}

int zonos3_generate(const zonos3_model& m, const zonos3_prompt& prompt, int max_frames, const zonos3_sampling& sp,
                    float cfg_scale, std::vector<int32_t>& out_codes, int& eos_frame, bool use_kv,
                    std::vector<int32_t>* out_full_ids, const zonos3_frame_cb& on_frame,
                    const zonos3_steering* steering, const zonos3_generate_features* features) {
    eos_frame = -1;
    if (!std::isfinite(cfg_scale) || cfg_scale < 0) {
        fprintf(stderr, "zonos3: speaker CFG scale must be finite and nonnegative\n");
        return -1;
    }
    const int W = (int)m.hp.n_codebooks + 1, ncb = (int)m.hp.n_codebooks;
    const int n0 = prompt.n_rows;
    std::vector<float> prompt_ids(prompt.ids.begin(), prompt.ids.end()), twin_ids;
    if (max_frames <= 0 || n0 <= 0 || n0 > (int)m.hp.n_ctx_train - max_frames || prompt.ids.size() != (size_t)n0 * W ||
        prompt.prefill_frames < 0 || prompt.prefill_frames > n0 || !valid_ids(m, prompt_ids.data(), n0)) {
        fprintf(stderr, "zonos3: prompt and generation exceed model context\n");
        return -1;
    }
    zonos3_prompt twin;
    const bool cfg = cfg_scale != 1.0f;
    if (cfg) {
        if (!zonos3_build_cfg_twin(m, prompt, twin))
            return -1;
        twin_ids.assign(twin.ids.begin(), twin.ids.end());
        // Twins share the request's accent adapter (the sgl scheduler copies accent_slot / accent_scale).
        twin.cond.accent = prompt.cond.accent;
        twin.cond.accent_strength = prompt.cond.accent_strength;
    }
    const size_t original = out_codes.size();
    int count;
    if (use_kv) {
        count = generate_kv(m, prompt_ids, prompt, twin_ids, cfg ? &twin : nullptr, cfg_scale, max_frames, sp,
                            out_codes, eos_frame, on_frame, steering, features);
    } else {
        if (steering && steering->hidden_active()) {
            fprintf(stderr, "zonos3: hidden-space steering needs the KV decode path (drop --recompute)\n");
            return -1;
        }
        if (features && (features->probe.enabled() || features->step_logits)) {
            fprintf(stderr, "zonos3: the alignment probe and step logits need the KV decode path (drop --recompute)\n");
            return -1;
        }
        count = generate_recompute(m, prompt_ids, prompt, twin_ids, cfg ? &twin : nullptr, cfg_scale, max_frames, sp,
                                   out_codes, eos_frame, on_frame);
    }
    if (count < 0) {
        out_codes.resize(original);
        return -1;
    }
    if (out_full_ids) {
        out_full_ids->clear();
        out_full_ids->reserve((size_t)(n0 + count) * W);
        out_full_ids->insert(out_full_ids->end(), prompt.ids.begin(), prompt.ids.end());
        for (int f = 0; f < count; ++f) {
            const int32_t* frame = out_codes.data() + original + (size_t)f * ncb;
            out_full_ids->insert(out_full_ids->end(), frame, frame + ncb);
            out_full_ids->push_back((int32_t)m.hp.text_vocab);
        }
    }
    return count;
}

zonos3_probe_readout zonos3_probe_reduce(const float* dist, int text_len) {
    zonos3_probe_readout r;
    if (!dist || text_len <= 0)
        return r;
    double total = 0.0, moment = 0.0;
    r.argmax = 0;
    for (int j = 0; j < text_len; ++j) {
        if (dist[j] > dist[r.argmax])
            r.argmax = j;
        total += dist[j];
        moment += (double)dist[j] * j;
    }
    r.centroid = (float)(moment / std::max(total, 1e-30));
    return r;
}

bool zonos3_locate_text_span(const zonos3_model& m, const int32_t* ids, int n_rows, int& start, int& length) {
    const int W = (int)m.hp.n_codebooks + 1, col = W - 1;
    constexpr int32_t BOS = 2, EOS = 3; // zonos3.tokenizer.bos / .eos (the loader requires 2 / 3)
    int bos = -1;
    for (int r = 0; r < n_rows && bos < 0; ++r)
        if (ids[(size_t)r * W + col] == BOS)
            bos = r;
    if (bos < 0)
        return false;
    for (int r = bos + 1; r < n_rows; ++r) {
        const int32_t t = ids[(size_t)r * W + col];
        if (t == EOS) {
            start = bos;
            length = r - bos + 1;
            return true;
        }
        if (t < 4)
            return false; // a special id strictly inside the span
    }
    return false;
}
