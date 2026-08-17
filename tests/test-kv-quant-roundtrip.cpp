// test-kv-quant-roundtrip.cpp — CRISPASR_KV_QUANT write/read round-trip.
//
// A quantised KV cache is written with `ggml_set_rows` in one graph and read
// back with `ggml_view_3d` + `ggml_cast(..., F32)` in a LATER graph. Nothing
// covered that: the only KV tests were core-cross-attn (F32 only) and
// flash-attn-defaults (flag plumbing), so every CRISPASR_KV_QUANT dtype was
// untested end to end.
//
// This test exists because a quantised KV cache was the leading suspect in an
// ark-asr empty-transcript hunt: `CRISPASR_KV_QUANT=q8_0` produced one token
// and then <im_end>, where q4_0 transcribed fine. It turned out NOT to be the
// cache — the ark prompt was missing upstream's instruction text, which left
// the first decode step marginal, and the KV dtype merely tipped it (the f16
// default failed too, at other clip lengths). See PLAN.md.
//
// The coverage is worth keeping regardless, and these assertions are what
// ruled the cache out: they are the reason that hunt moved on to the prompt
// instead of digging further into ggml. Two graphs on purpose — a decode step
// reads back rows a PREVIOUS graph wrote, so a single-graph probe of the same
// ops round-trips fine and proves nothing about the case that matters.
//
// Geometry is ark-asr's: head_dim 128, 2 KV heads. All six dtypes here are
// block-32, so a 128-wide row tiles exactly and a row-width mismatch cannot be
// what separates them. k-quants are deliberately absent — their 256-element
// blocks cannot tile a 128-wide row at all, which is why kv_dtype_parse
// refuses them rather than pretending.
//
// Tolerances are per-dtype quantisation error, not equality: the point is
// "the values survive the round-trip", not bit-exactness. A dtype that
// scrambles the cache misses by O(1), far outside any of these bounds.
#include <catch2/catch_test_macros.hpp>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint32_t nx() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (uint32_t)(s >> 11);
    }
    float sym() { return (float)((nx() % 20001) / 10000.0 - 1.0); } // [-1,1]
};

constexpr int kHeadDim = 128; // ark-asr llm.head_dim
constexpr int kKvHeads = 2;   // ark-asr llm.kv_heads
constexpr int kMaxCtx = 64;
constexpr int kWrite = 8; // rows written by the "prefill" graph

// Append `n` rows at row offset `at`, in a graph of their own.
void write_rows(ggml_tensor* cache, const float* vals, int n, int at, ggml_backend_t be) {
    std::vector<uint8_t> meta(4u * 1024 * 1024);
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* t_vals = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, n, kKvHeads);
    ggml_tensor* idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
    ggml_tensor* layer = ggml_view_3d(ctx, cache, kHeadDim, kMaxCtx, kKvHeads, cache->nb[1], cache->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, layer, t_vals, idx));

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_vals, vals, 0, (size_t)kHeadDim * n * kKvHeads * sizeof(float));
    std::vector<int64_t> ids((size_t)n);
    for (int i = 0; i < n; i++)
        ids[(size_t)i] = at + i;
    ggml_backend_tensor_set(idx, ids.data(), 0, ids.size() * sizeof(int64_t));
    ggml_backend_graph_compute(be, gf);
    ggml_gallocr_free(ga);
    ggml_free(ctx);
}

// Read the first `Lk` rows back the way core/attention.h does on a decode
// step: a strided per-layer view, dequantised to F32 with ggml_cast.
std::vector<float> read_rows(ggml_tensor* cache, int Lk, ggml_backend_t be) {
    std::vector<uint8_t> meta(4u * 1024 * 1024);
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* view = ggml_view_3d(ctx, cache, kHeadDim, Lk, kKvHeads, cache->nb[1], cache->nb[2], 0);
    ggml_tensor* deq =
        ggml_is_quantized(cache->type) ? ggml_cast(ctx, view, GGML_TYPE_F32) : ggml_cont(ctx, view);
    if (deq->type != GGML_TYPE_F32)
        deq = ggml_cast(ctx, deq, GGML_TYPE_F32);
    ggml_set_output(deq);
    ggml_build_forward_expand(gf, deq);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out((size_t)kHeadDim * Lk * kKvHeads);
    ggml_backend_tensor_get(deq, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return out;
}

// Write kWrite rows into a fresh cache of `type` in ONE graph, then read the
// whole layer back in a SECOND graph. Returns the dequantised [hd, kWrite,
// n_kv] block. Empty on allocation failure.
std::vector<float> roundtrip(ggml_type type, const std::vector<float>& src, ggml_backend_t be) {
    // ---- cache tensor, allocated exactly like the backends do (#367) ----
    ggml_init_params cp = {ggml_tensor_overhead() * 4 + 1024, nullptr, /*no_alloc=*/true};
    ggml_context* cctx = ggml_init(cp);
    ggml_tensor* cache = ggml_new_tensor_4d(cctx, type, kHeadDim, kMaxCtx, kKvHeads, 1);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, be);
    if (!cbuf) {
        ggml_free(cctx);
        return {};
    }
    ggml_backend_buffer_clear(cbuf, 0);

    std::vector<float> out;

    // ---- graph 1: set_rows write ----
    {
        std::vector<uint8_t> meta(4u * 1024 * 1024);
        ggml_init_params ip{meta.size(), meta.data(), true};
        ggml_context* ctx = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph(ctx);

        ggml_tensor* vals = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kWrite, kKvHeads);
        ggml_tensor* idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, kWrite);
        ggml_tensor* layer = ggml_view_3d(ctx, cache, kHeadDim, kMaxCtx, kKvHeads, cache->nb[1], cache->nb[2], 0);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, layer, vals, idx));

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(vals, src.data(), 0, src.size() * sizeof(float));
        std::vector<int64_t> ids(kWrite);
        for (int i = 0; i < kWrite; i++)
            ids[(size_t)i] = i;
        ggml_backend_tensor_set(idx, ids.data(), 0, ids.size() * sizeof(int64_t));
        ggml_backend_graph_compute(be, gf);
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }

    // ---- graph 2: view + cast read, the step-2 path ----
    {
        std::vector<uint8_t> meta(4u * 1024 * 1024);
        ggml_init_params ip{meta.size(), meta.data(), true};
        ggml_context* ctx = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph(ctx);

        ggml_tensor* view = ggml_view_3d(ctx, cache, kHeadDim, kWrite, kKvHeads, cache->nb[1], cache->nb[2], 0);
        ggml_tensor* deq = ggml_is_quantized(type) ? ggml_cast(ctx, view, GGML_TYPE_F32) : ggml_cont(ctx, view);
        if (deq->type != GGML_TYPE_F32)
            deq = ggml_cast(ctx, deq, GGML_TYPE_F32);
        ggml_set_output(deq);
        ggml_build_forward_expand(gf, deq);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_graph_compute(be, gf);

        out.resize((size_t)kHeadDim * kWrite * kKvHeads);
        ggml_backend_tensor_get(deq, out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }

    ggml_backend_buffer_free(cbuf);
    ggml_free(cctx);
    return out;
}

} // namespace

TEST_CASE("kv cache: every CRISPASR_KV_QUANT dtype survives a cross-graph round-trip", "[unit][kv-quant]") {
    struct Case {
        ggml_type type;
        const char* name;
        float tol; // max abs error for values in [-1, 1]
    };
    // f16 is the default and is exact enough to be a control: if it fails, the
    // harness is wrong rather than the dtype.
    const std::vector<Case> cases = {
        {GGML_TYPE_F16, "f16", 1e-3f},   {GGML_TYPE_Q8_0, "q8_0", 0.02f}, {GGML_TYPE_Q5_1, "q5_1", 0.10f},
        {GGML_TYPE_Q5_0, "q5_0", 0.10f}, {GGML_TYPE_Q4_1, "q4_1", 0.20f}, {GGML_TYPE_Q4_0, "q4_0", 0.20f},
    };

    Rng rng(0x370ULL);
    std::vector<float> src((size_t)kHeadDim * kWrite * kKvHeads);
    for (auto& x : src)
        x = rng.sym();

    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);

    for (const auto& c : cases) {
        INFO("dtype = " << c.name);
        const std::vector<float> got = roundtrip(c.type, src, be);
        REQUIRE(got.size() == src.size());

        double worst = 0.0;
        size_t worst_at = 0;
        for (size_t i = 0; i < src.size(); i++) {
            const double e = std::fabs((double)got[i] - (double)src[i]);
            if (e > worst) {
                worst = e;
                worst_at = i;
            }
        }
        INFO("worst |err| = " << worst << " at index " << worst_at << " (wrote " << src[worst_at] << ", read back "
                              << got[worst_at] << ")");
        REQUIRE(worst <= (double)c.tol);
    }

    ggml_backend_free(be);
}

// The shape a real decode takes: prefill writes T rows, a decode step appends
// ONE row at n_past, and the next forward pass reads the grown history. That
// second pass is where an empty transcript would surface, so the
// append-then-read-back sequence is the part worth pinning, not just a single
// bulk write.
TEST_CASE("kv cache: appended decode rows read back correctly for every dtype", "[unit][kv-quant]") {
    struct Case {
        ggml_type type;
        const char* name;
        float tol;
    };
    const std::vector<Case> cases = {
        {GGML_TYPE_F16, "f16", 1e-3f},   {GGML_TYPE_Q8_0, "q8_0", 0.02f}, {GGML_TYPE_Q5_1, "q5_1", 0.10f},
        {GGML_TYPE_Q5_0, "q5_0", 0.10f}, {GGML_TYPE_Q4_1, "q4_1", 0.20f}, {GGML_TYPE_Q4_0, "q4_0", 0.20f},
    };

    Rng rng(0x371ULL);
    std::vector<float> prefill((size_t)kHeadDim * kWrite * kKvHeads);
    for (auto& x : prefill)
        x = rng.sym();
    std::vector<float> step((size_t)kHeadDim * 1 * kKvHeads);
    for (auto& x : step)
        x = rng.sym();

    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);

    const int Lk = kWrite + 1;
    for (const auto& c : cases) {
        INFO("dtype = " << c.name);

        ggml_init_params cp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
        ggml_context* cctx = ggml_init(cp);
        ggml_tensor* cache = ggml_new_tensor_4d(cctx, c.type, kHeadDim, kMaxCtx, kKvHeads, 1);
        ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, be);
        REQUIRE(cbuf != nullptr);
        ggml_backend_buffer_clear(cbuf, 0);

        write_rows(cache, prefill.data(), kWrite, 0, be);  // prefill
        write_rows(cache, step.data(), 1, kWrite, be);     // one decode step
        const std::vector<float> got = read_rows(cache, Lk, be);
        REQUIRE(got.size() == (size_t)kHeadDim * Lk * kKvHeads);

        double worst = 0.0;
        int worst_row = -1;
        for (int h = 0; h < kKvHeads; h++) {
            for (int r = 0; r < Lk; r++) {
                for (int d = 0; d < kHeadDim; d++) {
                    const float want = (r < kWrite) ? prefill[(size_t)(h * kWrite + r) * kHeadDim + (size_t)d]
                                                    : step[(size_t)h * kHeadDim + (size_t)d];
                    const double e = std::fabs((double)got[(size_t)(h * Lk + r) * kHeadDim + (size_t)d] - (double)want);
                    if (e > worst) {
                        worst = e;
                        worst_row = r;
                    }
                }
            }
        }
        INFO("worst |err| = " << worst << " at row " << worst_row << " (appended row is " << kWrite << ")");
        REQUIRE(worst <= (double)c.tol);

        ggml_backend_buffer_free(cbuf);
        ggml_free(cctx);
    }

    ggml_backend_free(be);
}
