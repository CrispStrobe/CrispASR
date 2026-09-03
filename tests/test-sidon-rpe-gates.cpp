// test-sidon-rpe-gates.cpp — unit tests for src/sidon_rpe_gates.h, the
// length-aware relative-position-bias selection (#416 follow-up).
//
// Mirrors tests/test-htdemucs-gates.cpp and tests/test-mel-band-gates.cpp. The
// AUTO defaults encode a measured fact: on the #416 reporter's GTX 1660 SUPER
// (Vulkan, T=557, same file and binary so the RPE mode is the only variable)
// the predictor ran 213.80 ms under `expand` vs 575.25 ms under the
// `bucket-direct` default — 2.7x. AUTO takes that speedup exactly when
// expand's extra transient footprint fits the budget.
//
// These cases lock the whole table, including the two properties that matter
// most: an explicit CRISPASR_SIDON_RPE is NEVER auto-overridden (it is the
// #416 bisection handle), and AUTO never changes CPU behaviour, because the
// 2.7x is one device's measurement and nothing supports generalising it.

#include "sidon_rpe_gates.h"

#include <catch2/catch_test_macros.hpp>

using sidon_rpe_gates::expand_extra_bytes;
using sidon_rpe_gates::mode;
using sidon_rpe_gates::resolve;

namespace {
// The shipped model: hidden 1024, heads 16 -> head_dim 64.
constexpr int HD = 64;
constexpr int H = 16;

// 196*T^2 <= 256 MiB  =>  T <= 1170.
constexpr int T_FITS = 1170;
constexpr int T_OVER = 1171;
} // namespace

TEST_CASE("sidon RPE gates: AUTO on GPU at a short length picks expand", "[unit][sidon]") {
    const auto r = resolve(nullptr, nullptr, 557, HD, H, /*is_gpu=*/true);
    REQUIRE(r.m == mode::expand);
    REQUIRE(r.auto_expand);
    REQUIRE_FALSE(r.from_env);
}

TEST_CASE("sidon RPE gates: AUTO on GPU past the budget falls back to bucket_direct", "[unit][sidon]") {
    const auto r = resolve(nullptr, nullptr, 3000, HD, H, true);
    REQUIRE(r.m == mode::bucket_direct);
    REQUIRE_FALSE(r.auto_expand);
}

TEST_CASE("sidon RPE gates: the budget boundary is exact", "[unit][sidon]") {
    REQUIRE(resolve(nullptr, nullptr, T_FITS, HD, H, true).m == mode::expand);
    REQUIRE(resolve(nullptr, nullptr, T_OVER, HD, H, true).m == mode::bucket_direct);
    // and the formula the boundary is derived from
    REQUIRE(expand_extra_bytes(T_FITS, HD, H) <= 256ull * 1024 * 1024);
    REQUIRE(expand_extra_bytes(T_OVER, HD, H) > 256ull * 1024 * 1024);
}

TEST_CASE("sidon RPE gates: AUTO never changes CPU behaviour", "[unit][sidon]") {
    // Every length, including ones that fit the budget comfortably.
    for (const int T : {1, 100, 557, 1170, 3000}) {
        const auto r = resolve(nullptr, nullptr, T, HD, H, /*is_gpu=*/false);
        REQUIRE(r.m == mode::bucket_direct);
        REQUIRE_FALSE(r.auto_expand);
    }
}

TEST_CASE("sidon RPE gates: an explicit mode is honoured exactly and never auto-overridden", "[unit][sidon]") {
    // expand forced at a length AUTO would have refused
    const auto e = resolve("expand", nullptr, 3000, HD, H, true);
    REQUIRE(e.m == mode::expand);
    REQUIRE(e.from_env);
    REQUIRE_FALSE(e.auto_expand); // it came from the env, not from AUTO

    // bucket-direct forced at a length AUTO would have upgraded (#416 handle)
    const auto b = resolve("bucket-direct", nullptr, 557, HD, H, true);
    REQUIRE(b.m == mode::bucket_direct);
    REQUIRE(b.from_env);

    // bucket forced, on CPU and GPU alike
    REQUIRE(resolve("bucket", nullptr, 557, HD, H, true).m == mode::bucket);
    REQUIRE(resolve("bucket", nullptr, 557, HD, H, false).m == mode::bucket);
}

TEST_CASE("sidon RPE gates: an unrecognised value falls back to bucket_direct and is reported", "[unit][sidon]") {
    const auto r = resolve("nonsense", nullptr, 557, HD, H, true);
    REQUIRE(r.m == mode::bucket_direct);
    REQUIRE(r.from_env); // explicit-but-bad still counts as explicit: no AUTO upgrade
    REQUIRE(sidon_rpe_gates::env_mode_is_unknown("nonsense"));
    REQUIRE_FALSE(sidon_rpe_gates::env_mode_is_unknown("expand"));
    REQUIRE_FALSE(sidon_rpe_gates::env_mode_is_unknown(nullptr));
    REQUIRE_FALSE(sidon_rpe_gates::env_mode_is_unknown(""));
}

TEST_CASE("sidon RPE gates: an empty env value is AUTO, not an explicit choice", "[unit][sidon]") {
    const auto r = resolve("", nullptr, 557, HD, H, true);
    REQUIRE(r.m == mode::expand);
    REQUIRE(r.auto_expand);
    REQUIRE_FALSE(r.from_env);
}

TEST_CASE("sidon RPE gates: budget 0 disables AUTO entirely", "[unit][sidon]") {
    const auto r = resolve(nullptr, "0", 557, HD, H, true);
    REQUIRE(r.m == mode::bucket_direct);
    REQUIRE_FALSE(r.auto_expand);
    // negative is clamped to 0, not treated as unlimited
    REQUIRE(resolve(nullptr, "-1", 557, HD, H, true).m == mode::bucket_direct);
}

TEST_CASE("sidon RPE gates: the budget override moves the threshold both ways", "[unit][sidon]") {
    // 58 MiB at T=557: a 64 MiB budget admits it, a 32 MiB budget does not.
    REQUIRE(resolve(nullptr, "64", 557, HD, H, true).m == mode::expand);
    REQUIRE(resolve(nullptr, "32", 557, HD, H, true).m == mode::bucket_direct);
    // and a large budget admits what the default refused
    REQUIRE(resolve(nullptr, "4096", 3000, HD, H, true).m == mode::expand);
}

TEST_CASE("sidon RPE gates: expand_extra_bytes matches the documented layout", "[unit][sidon]") {
    // 4 * T^2 * (hd + 1 - H)
    REQUIRE(expand_extra_bytes(557, HD, H) == 4ull * 49 * 557 * 557);
    REQUIRE(expand_extra_bytes(1024, HD, H) == 4ull * 49 * 1024 * 1024);

    // Degenerate shapes: expand is not the heavier path, and nothing is negative.
    REQUIRE(expand_extra_bytes(557, 8, 16) == 0);
    REQUIRE(expand_extra_bytes(557, 15, 16) == 0);
    REQUIRE(expand_extra_bytes(0, HD, H) == 0);
    REQUIRE(expand_extra_bytes(-5, HD, H) == 0);
    REQUIRE(expand_extra_bytes(557, 0, H) == 0);
    REQUIRE(expand_extra_bytes(557, HD, 0) == 0);

    // Absurd T saturates rather than wrapping — a wrap would read as "tiny"
    // and hand AUTO a multi-terabyte allocation.
    REQUIRE(expand_extra_bytes(2000000000, HD, H) == UINT64_MAX);
    REQUIRE(resolve(nullptr, nullptr, 2000000000, HD, H, true).m == mode::bucket_direct);
}
