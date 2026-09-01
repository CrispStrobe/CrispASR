// test-htdemucs-gates.cpp — unit tests for src/htdemucs_gates.h (#413/#414).
//
// The AUTO defaults encode measured facts: fused-graph-on-GPU ~20x faster
// than CPU/BLAS; per-layer graphs (GPU or CPU) SLOWER than BLAS. These cases
// lock the whole decision table, including the two review catches from
// PR #414: (a) FUSED=1 alone must not select the slow per-layer GPU path —
// it implies the graph it needs; (b) CRISPASR_HTDEMUCS_GPU=0 must actually
// opt out even though the CLI's params.use_gpu defaults to true.

#include "htdemucs_gates.h"

#include <catch2/catch_test_macros.hpp>

using htdemucs_gates::resolve;

TEST_CASE("htdemucs gates: AUTO on a real-GPU host = fused graph on GPU", "[unit][htdemucs]") {
    const auto r = resolve(nullptr, nullptr, nullptr, /*caller_use_gpu=*/true, /*have_real_gpu=*/true);
    REQUIRE(r.use_graph);
    REQUIRE(r.use_fused);
    REQUIRE(r.gpu_backend);
}

TEST_CASE("htdemucs gates: AUTO on a CPU-only host = legacy BLAS, unchanged", "[unit][htdemucs]") {
    const auto r = resolve(nullptr, nullptr, nullptr, true, /*have_real_gpu=*/false);
    REQUIRE_FALSE(r.use_graph);
    REQUIRE_FALSE(r.use_fused);
    REQUIRE_FALSE(r.gpu_backend);
}

TEST_CASE("htdemucs gates: GPU=0 opts out even though the CLI defaults use_gpu=true", "[unit][htdemucs]") {
    const auto r = resolve("0", nullptr, nullptr, /*caller_use_gpu=*/true, true);
    REQUIRE_FALSE(r.gpu_backend);
    REQUIRE_FALSE(r.use_graph); // AUTO graph follows gpu -> BLAS
}

TEST_CASE("htdemucs gates: --no-gpu (caller intent) keeps BLAS; GPU=1 env overrides it", "[unit][htdemucs]") {
    const auto no_gpu = resolve(nullptr, nullptr, nullptr, /*caller_use_gpu=*/false, true);
    REQUIRE_FALSE(no_gpu.gpu_backend);
    REQUIRE_FALSE(no_gpu.use_graph);

    const auto forced = resolve("1", nullptr, nullptr, /*caller_use_gpu=*/false, true);
    REQUIRE(forced.gpu_backend);
    REQUIRE(forced.use_graph);
    REQUIRE(forced.use_fused);
}

TEST_CASE("htdemucs gates: GPU=1 on a GPU-less host still resolves to CPU/BLAS", "[unit][htdemucs]") {
    const auto r = resolve("1", nullptr, nullptr, true, /*have_real_gpu=*/false);
    REQUIRE_FALSE(r.gpu_backend);
    REQUIRE_FALSE(r.use_graph);
}

TEST_CASE("htdemucs gates: explicit GGML=1 keeps the old opt-in behavior on CPU", "[unit][htdemucs]") {
    // CPU graph path, per-layer (FUSED still AUTO-off off-GPU) — exactly the
    // pre-#414 CRISPASR_HTDEMUCS_GGML=1 semantics.
    const auto r = resolve(nullptr, "1", nullptr, true, false);
    REQUIRE(r.use_graph);
    REQUIRE_FALSE(r.use_fused);
    REQUIRE_FALSE(r.gpu_backend);
}

TEST_CASE("htdemucs gates: FUSED=1 alone implies the graph it needs (#414 review catch)", "[unit][htdemucs]") {
    const auto gpu = resolve(nullptr, nullptr, "1", true, true);
    REQUIRE(gpu.use_graph);
    REQUIRE(gpu.use_fused);
    REQUIRE(gpu.gpu_backend);

    // On CPU it stays inert-but-consistent: graph engages, fused engages,
    // CPU backend — the explicit-CPU-fused A/B configuration.
    const auto cpu = resolve(nullptr, nullptr, "1", true, false);
    REQUIRE(cpu.use_graph);
    REQUIRE(cpu.use_fused);
    REQUIRE_FALSE(cpu.gpu_backend);
}

TEST_CASE("htdemucs gates: GGML=0 explicitly forces BLAS even on GPU", "[unit][htdemucs]") {
    const auto r = resolve(nullptr, "0", nullptr, true, true);
    REQUIRE_FALSE(r.use_graph);
    REQUIRE_FALSE(r.use_fused);
    REQUIRE_FALSE(r.gpu_backend);
}

TEST_CASE("htdemucs gates: FUSED=0 on GPU keeps graph+GPU but unfused (bisection arm)", "[unit][htdemucs]") {
    const auto r = resolve(nullptr, nullptr, "0", true, true);
    REQUIRE(r.use_graph);
    REQUIRE_FALSE(r.use_fused);
    REQUIRE(r.gpu_backend);
}
