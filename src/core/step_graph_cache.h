#pragma once
// Cached, Lk-bucketed single-token decode graphs for autoregressive backends.
//
// Generalises the funasr step-graph cache (58e7b7ea) so the next AR decoder
// does not have to re-derive it. funasr keeps its own copy for now; this
// header is the version new callers should wire to.
//
// WHAT IT BUYS. An AR decode step rebuilds an identical graph every token and
// re-plans the scheduler around it. Building once per Lk bucket and driving
// gallocr + the backend directly removes that: funasr measured the per-step
// `decode_step_prep` line drop from 2.1 ms/tok to 0.15 ms/tok.
//
//     THE TRAP THIS EXISTS TO AVOID.
// The obvious way to keep topology constant is one graph at `max_ctx`. That
// is a LOSS, not a win: every step then reads the whole max_ctx KV window
// through flash-attn instead of the n_past+1 rows that hold data. funasr
// measured 359 ms/tok against 212 ms/tok for the per-call path it replaced —
// +69%. Graph-prep savings are a constant ~2 ms; wasted-KV cost scales with
// (Lk - n_past) x layers x head_dim and dominates immediately. Round Lk up to
// a NARROW stride, never to a power of two and never to max_ctx.
//
//     MEASURED ON VOXTRAL-3B: NO WIN. THE LEVER IS MODEL-SIZE-DEPENDENT.
// Kaggle A/B (chr1s4/crispasr-voxtral-stepcache-ab v3, q4_k, 4 CPU threads,
// decode time isolated via CRISPASR_VOXTRAL_BENCH, best of 3):
//
//     arm                     decode ms   ms/step
//     cache off                    4934     189.8
//     cache on, width 16           4993     192.0   <- 1.2% SLOWER
//     cache on, width >= max_ctx  17767     683.3   <- 3.6x slower
//
// Output was bit-identical across all three. The saving this cache offers is
// ~2 ms of graph prep per step; a voxtral decode step costs ~190 ms, so the
// prep is ~1% of it and the bucket's own mask build + gallocr alloc costs
// slightly more than it saves. funasr's win was real because its step is far
// cheaper -- the SAME 2 ms is a large share there. So the question is not
// "which width" but "is prep a meaningful fraction of a step at all", and for
// a 3B/30-layer/d=3072 decoder it is not. Do not wire this into another large
// AR decoder expecting a speed-up; measure the step cost first.
//
// The max_ctx row is the useful half: it confirms the trap below on a second
// model, and at 3.6x it is far worse here than the +69% funasr measured.
//
//     THE WIDTH IS HARDWARE- AND MODEL-SPECIFIC. MEASURE IT.
// A bucket of width w wastes w/2 keys on average. Break-even is
// w ~= prep_ms / per_key_ms. funasr measured per_key_ms ~= 0.28 on x86 CPU
// (q4_k, 16 heads, head_dim 128, 28 layers) giving w ~= 15, and picked 16;
// w=32 measured 223 ms/tok and w=64 232, both net losses there. A model with
// different head geometry, or a platform where graph construction is dearer
// relative to memory bandwidth (M1 was quoted at ~30 ms/tok prep), moves
// break-even. kDefaultWidth is therefore the CONSERVATIVE pick that cannot
// lose much, not a tuned constant — callers should expose an env override and
// measure on their own arms before claiming a number.
//
// BIT-IDENTITY. Padding a bucket is exact, not approximate: slots in
// [n_past+1, Lk) are masked -inf, ggml's flash-attn skips a fully masked kv
// position outright, and soft_max_ext maps -inf to a zero weight, so neither
// the running max nor the accumulator observes them. Output is bit-identical
// to an Lk = n_past+1 graph. (This holds for ATTENTION padding only. It does
// NOT transfer to padded ENCODER buckets whenever the stack contains
// time-axis convolutions, which leak padding across frames.)
//
// SCHED IS BYPASSED ON PURPOSE. Cached graphs hold views into the persistent
// KV buffers; ggml_backend_sched re-plans per call and fights that. The
// cached path drives ggml_gallocr + ggml_backend_graph_compute on a single
// backend instead, which is why `supports_graph` below must gate it: without
// the sched there is no CPU fallback for an unsupported op.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace core_step_cache {

// Conservative default; see the header note on measuring it.
constexpr int kDefaultWidth = 16;

// Keep at most this many live arenas. Lk only grows inside one decode, so an
// entry is dead as soon as the next bucket is built; the depth exists so a
// second, shorter utterance can still reuse an earlier bucket.
constexpr size_t kDefaultMaxLive = 4;

// Round the required KV window up to the next `width`-wide bucket, clamped to
// max_ctx. Returns -1 when the request cannot be served at all.
inline int bucket_for(int need, int width, int max_ctx) {
    if (need <= 0 || max_ctx <= 0 || need > max_ctx)
        return -1;
    const int w = width > 0 ? width : kDefaultWidth;
    long long lk = ((long long)(need + w - 1) / w) * w;
    if (lk > max_ctx)
        lk = max_ctx;
    return (int)lk;
}

// Every op in `gf` must run on `be`, because the cached path has no sched to
// fall back to. Reports the first offender by name — a silent `false` here
// would strand a caller with an unexplained slow path.
inline bool supports_graph(ggml_backend_t be, ggml_cgraph* gf, const char* tag) {
    if (!be || !gf)
        return false;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor* node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_NONE)
            continue;
        if (!ggml_backend_supports_op(be, node)) {
            std::fprintf(stderr, "%s: step-graph cache off — op %s unsupported on %s; using sched path\n", tag,
                         ggml_op_desc(node), ggml_backend_name(be));
            return false;
        }
    }
    return true;
}

// Fill a [Lk] F16 decode mask: real keys [0, n_past] pass, bucket padding
// [n_past+1, Lk) is -inf. See the bit-identity note above.
inline void fill_decode_mask(std::vector<ggml_fp16_t>& mask, int n_past, int Lk) {
    mask.resize((size_t)Lk);
    const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
    for (int k = 0; k < Lk; k++)
        mask[(size_t)k] = (k <= n_past) ? zero_h : ninf_h;
}

struct Entry {
    int lk = 0;
    std::vector<uint8_t> meta; // arena backing ctx0; must outlive the graph
    ggml_context* ctx0 = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
};

// Non-copyable owner of the live entries. One per model context.
struct Cache {
    bool enabled = true;
    int width = kDefaultWidth;
    size_t max_live = kDefaultMaxLive;
    int n_builds = 0; // bench: how many distinct buckets were compiled
    std::vector<Entry> entries;

    Cache() = default;
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    ~Cache() { clear(); }

    void clear() {
        for (auto& e : entries) {
            if (e.galloc)
                ggml_gallocr_free(e.galloc);
            if (e.ctx0)
                ggml_free(e.ctx0);
        }
        entries.clear();
    }

    // Fetch (building on first use) the entry whose fixed Lk covers `need`
    // keys. `build` receives the entry's own arena and the bucket's Lk and
    // must return a graph built entirely inside that arena.
    //
    // Returns nullptr when disabled, out of range, or on a build/reserve
    // failure — in which case the cache disables itself so the caller's
    // per-call fallback is taken from then on rather than retried per token.
    template <typename BuildFn>
    Entry* get_or_build(ggml_backend_t be, int need, int max_ctx, size_t n_nodes, const char* tag, BuildFn build) {
        if (!enabled || !be)
            return nullptr;
        const int lk = bucket_for(need, width, max_ctx);
        if (lk < 0)
            return nullptr;

        for (auto& e : entries) {
            if (e.lk == lk)
                return &e;
        }

        while (entries.size() >= max_live) {
            Entry& victim = entries.front();
            if (victim.galloc)
                ggml_gallocr_free(victim.galloc);
            if (victim.ctx0)
                ggml_free(victim.ctx0);
            entries.erase(entries.begin());
        }

        Entry e;
        e.lk = lk;
        e.meta.resize(ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false));
        ggml_init_params ip = {e.meta.size(), e.meta.data(), /*no_alloc=*/true};
        e.ctx0 = ggml_init(ip);

        // Name the actual reason. A single "build/reserve failed" for all
        // three causes would send the next reader to the wrong place —
        // an unsupported op is a backend-capability fact, not a build bug.
        const char* why = nullptr;
        if (!e.ctx0) {
            why = "arena init failed";
        } else if (!(e.graph = build(e.ctx0, lk))) {
            why = "graph build returned null";
        } else if (!supports_graph(be, e.graph, tag)) {
            why = "backend cannot run every op (reported above)";
        } else {
            e.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!e.galloc) {
                why = "gallocr alloc failed";
            } else if (!ggml_gallocr_reserve(e.galloc, e.graph)) {
                why = "gallocr reserve failed";
            } else {
                n_builds++;
                // Positive trace. Without it an A/B cannot tell "the cache ran
                // and matched" from "the cache silently fell back and the two
                // arms were the same code" — which look identical in both the
                // transcript and the timing, and the second reads as success.
                std::fprintf(stderr, "%s: step-graph cache ACTIVE — built bucket Lk=%d (width=%d, %d built)\n", tag, lk,
                             width > 0 ? width : kDefaultWidth, n_builds);
                entries.push_back(std::move(e));
                return &entries.back();
            }
        }

        std::fprintf(stderr, "%s: step-graph cache disabled at Lk=%d — %s; using the per-call path\n", tag, lk, why);
        if (e.galloc)
            ggml_gallocr_free(e.galloc);
        if (e.ctx0)
            ggml_free(e.ctx0);
        enabled = false;
        return nullptr;
    }
};

} // namespace core_step_cache
