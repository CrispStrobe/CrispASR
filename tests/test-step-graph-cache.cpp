// Unit tests for core/step_graph_cache.h — the Lk-bucket picker and the
// decode mask that together carry the cache's bit-identity claim.
//
// These two functions are the whole correctness surface reachable without a
// model: get_or_build needs a live backend and a real graph, so it is proved
// on Kaggle by an A/B that must emit IDENTICAL token ids, not here.
//
// What is worth asserting here is the part that silently degrades instead of
// crashing. A bucket picker that quietly returns max_ctx still produces
// correct transcripts — it just hands back the entire +69% regression the
// bucketing exists to avoid, and no output would look wrong. So the tests
// below pin the waste, not just the validity.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "core/step_graph_cache.h"

using namespace core_step_cache;

TEST_CASE("bucket_for rounds up to the stride, never past max_ctx", "[unit][core][stepcache]") {
    // Exact multiples stay put — no wasted key at all.
    REQUIRE(bucket_for(16, 16, 4096) == 16);
    REQUIRE(bucket_for(512, 16, 4096) == 512);

    // Anything else goes to the next stride, not the next power of two.
    REQUIRE(bucket_for(1, 16, 4096) == 16);
    REQUIRE(bucket_for(17, 16, 4096) == 32);
    REQUIRE(bucket_for(210, 16, 4096) == 224); // a coarse ladder would say 256
    REQUIRE(bucket_for(530, 16, 4096) == 544); // a coarse ladder would say 1024

    // Clamped at max_ctx rather than overshooting the allocated cache.
    REQUIRE(bucket_for(4090, 16, 4096) == 4096);
    REQUIRE(bucket_for(4096, 16, 4096) == 4096);
}

TEST_CASE("bucket_for refuses what the cache cannot serve", "[unit][core][stepcache]") {
    REQUIRE(bucket_for(4097, 16, 4096) == -1); // beyond the KV cache
    REQUIRE(bucket_for(0, 16, 4096) == -1);
    REQUIRE(bucket_for(-1, 16, 4096) == -1);
    REQUIRE(bucket_for(16, 16, 0) == -1); // kv_init has not run
}

TEST_CASE("bucket width bounds the wasted KV window", "[unit][core][stepcache]") {
    // The property that actually matters: whatever the width, a bucket never
    // wastes a full width. This is what separates fine bucketing from the
    // fixed-at-max_ctx design that measured +69% decode CPU on funasr.
    for (int w : {1, 8, 16, 32, 64, 128}) {
        for (int need = 1; need <= 1200; need++) {
            const int lk = bucket_for(need, w, 4096);
            REQUIRE(lk >= need);
            REQUIRE(lk - need < w);
        }
    }
}

TEST_CASE("width <= 0 falls back to the documented default", "[unit][core][stepcache]") {
    REQUIRE(bucket_for(17, 0, 4096) == bucket_for(17, kDefaultWidth, 4096));
    REQUIRE(bucket_for(17, -5, 4096) == bucket_for(17, kDefaultWidth, 4096));
}

TEST_CASE("fill_decode_mask passes exactly the written keys", "[unit][core][stepcache]") {
    const int n_past = 10;
    const int Lk = 32;
    std::vector<ggml_fp16_t> m;
    fill_decode_mask(m, n_past, Lk);

    REQUIRE(m.size() == (size_t)Lk);
    for (int k = 0; k < Lk; k++) {
        const float v = ggml_fp16_to_fp32(m[(size_t)k]);
        if (k <= n_past) {
            // Real key: must contribute unmodified. A nonzero bias here would
            // silently reweight attention rather than fail.
            REQUIRE(v == 0.0f);
        } else {
            // Padding slot: must be -inf, not merely very negative. soft_max
            // maps -inf to an exactly-zero weight; a large finite value leaks
            // a small one, which is the difference between bit-identical and
            // "close enough" — and only the former is claimed.
            REQUIRE(std::isinf(v));
            REQUIRE(v < 0.0f);
        }
    }
}

TEST_CASE("a fully-tight bucket has no padding at all", "[unit][core][stepcache]") {
    std::vector<ggml_fp16_t> m;
    fill_decode_mask(m, 15, 16); // n_past+1 == Lk
    REQUIRE(m.size() == 16);
    for (auto h : m)
        REQUIRE(ggml_fp16_to_fp32(h) == 0.0f);
}

TEST_CASE("the mask reflects n_past, so a stale one would be caught", "[unit][core][stepcache]") {
    // Positive control for the assertions above: they must be able to FAIL.
    // Two different n_past in the same bucket have to yield different masks —
    // if fill_decode_mask ignored n_past (e.g. by caching), every test here
    // would still pass while the cached graph attended to stale KV rows.
    std::vector<ggml_fp16_t> a, b;
    fill_decode_mask(a, 4, 32);
    fill_decode_mask(b, 5, 32);
    REQUIRE(a.size() == b.size());
    REQUIRE(a != b);
    REQUIRE(ggml_fp16_to_fp32(a[5]) != ggml_fp16_to_fp32(b[5]));
}
