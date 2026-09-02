// test-mel-band-gates.cpp — unit tests for src/mel_band_gates.h (Change 176).
//
// Mirrors tests/test-htdemucs-gates.cpp (PR #414): the AUTO defaults encode
// measured facts — fused single graph on GPU ~112x faster than the per-layer
// graphs (2.6 s vs 4:47 on a 30 s clip, RTX 3090 Ti) and RTF ~0.09 on a full
// song; the legacy CPU path is RTF ~57. These cases lock the whole decision
// table, including the #414 review catches: (a) FUSED=1 alone must not leave
// the graph off — it implies the graph it needs; (b) CRISPASR_MELBAND_GPU=0
// must actually opt out even though the CLI's params.use_gpu can be true.

#include "mel_band_gates.h"

#include <catch2/catch_test_macros.hpp>

using mel_band_gates::resolve;

TEST_CASE("mel-band gates: AUTO on a real-GPU host = fused graph on GPU", "[unit][mel-band]") {
    const auto r = resolve(nullptr, nullptr, nullptr, /*caller_use_gpu=*/true, /*have_real_gpu=*/true);
    REQUIRE(r.use_graph);
    REQUIRE(r.use_fused);
    REQUIRE(r.gpu_backend);
}

TEST_CASE("mel-band gates: AUTO on a CPU-only host = legacy CPU path, unchanged", "[unit][mel-band]") {
    const auto r = resolve(nullptr, nullptr, nullptr, true, /*have_real_gpu=*/false);
    REQUIRE_FALSE(r.use_graph);
    REQUIRE_FALSE(r.use_fused);
    REQUIRE_FALSE(r.gpu_backend);
}

TEST_CASE("mel-band gates: GPU=0 opts out even though the caller may default use_gpu=true", "[unit][mel-band]") {
    const auto r = resolve("0", nullptr, nullptr, /*caller_use_gpu=*/true, true);
    REQUIRE_FALSE(r.gpu_backend);
    REQUIRE_FALSE(r.use_graph); // AUTO graph follows gpu -> CPU path
}

TEST_CASE("mel-band gates: caller intent (use_gpu=false) keeps CPU; GPU=1 env overrides it", "[unit][mel-band]") {
    const auto no_gpu = resolve(nullptr, nullptr, nullptr, /*caller_use_gpu=*/false, true);
    REQUIRE_FALSE(no_gpu.gpu_backend);
    REQUIRE_FALSE(no_gpu.use_graph);

    const auto forced = resolve("1", nullptr, nullptr, /*caller_use_gpu=*/false, true);
    REQUIRE(forced.gpu_backend);
    REQUIRE(forced.use_graph);
    REQUIRE(forced.use_fused);
}

TEST_CASE("mel-band gates: GPU=1 on a GPU-less host still resolves to the CPU path", "[unit][mel-band]") {
    const auto r = resolve("1", nullptr, nullptr, true, /*have_real_gpu=*/false);
    REQUIRE_FALSE(r.gpu_backend);
    REQUIRE_FALSE(r.use_graph);
}

TEST_CASE("mel-band gates: explicit GGML=1 keeps the opt-in graph path on CPU (unfused)", "[unit][mel-band]") {
    // CPU graph path, per-layer (FUSED still AUTO-off off-GPU) — the explicit
    // A/B configuration used during Change 176 parity runs.
    const auto r = resolve(nullptr, "1", nullptr, true, false);
    REQUIRE(r.use_graph);
    REQUIRE_FALSE(r.use_fused);
    REQUIRE_FALSE(r.gpu_backend);
}

TEST_CASE("mel-band gates: FUSED=1 alone implies the graph it needs (#414 review catch)", "[unit][mel-band]") {
    const auto gpu = resolve(nullptr, nullptr, "1", true, true);
    REQUIRE(gpu.use_graph);
    REQUIRE(gpu.use_fused);
    REQUIRE(gpu.gpu_backend);

    // On CPU it stays consistent: graph engages, fused engages, CPU backend.
    const auto cpu = resolve(nullptr, nullptr, "1", true, false);
    REQUIRE(cpu.use_graph);
    REQUIRE(cpu.use_fused);
    REQUIRE_FALSE(cpu.gpu_backend);
}

TEST_CASE("mel-band gates: GGML=0 explicitly forces the CPU path even on GPU", "[unit][mel-band]") {
    const auto r = resolve(nullptr, "0", nullptr, true, true);
    REQUIRE_FALSE(r.use_graph);
    REQUIRE_FALSE(r.use_fused);
    REQUIRE_FALSE(r.gpu_backend);
}

TEST_CASE("mel-band gates: FUSED=0 on GPU keeps graph+GPU but unfused (bisection arm)", "[unit][mel-band]") {
    const auto r = resolve(nullptr, nullptr, "0", true, true);
    REQUIRE(r.use_graph);
    REQUIRE_FALSE(r.use_fused);
    REQUIRE(r.gpu_backend);
}
