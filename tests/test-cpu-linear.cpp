// core_cpu_linear: the cblas_sgemm path must match the scalar loop (PR #611,
// FireRed VAD on OpenBLAS), and a build that found CPU BLAS must actually
// compile the BLAS path in (the #611 bug was a silently scalar build).
#include "core/cpu_linear.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

#ifndef CRISPASR_EXPECT_CPU_BLAS
#define CRISPASR_EXPECT_CPU_BLAS 0
#endif

namespace {

struct Case {
    int T, K, N;
    bool bias;
};

double max_rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++)
        m = std::fmax(m, std::fabs(a[i] - b[i]) / (1.0 + std::fabs(b[i])));
    return m;
}

} // namespace

TEST_CASE("cpu_linear: BLAS is compiled in exactly when CMake found it", "[unit][cpu_linear]") {
    REQUIRE(core_cpu_linear::have_blas() == (CRISPASR_EXPECT_CPU_BLAS != 0));
}

TEST_CASE("cpu_linear: BLAS path matches the scalar loop", "[unit][cpu_linear]") {
    // FireRed VAD DFSMN shapes (T frames x 80/140/256 dims) plus odd edges.
    const Case cases[] = {{1, 1, 1, true},       {1, 80, 256, true}, {7, 256, 140, false}, {64, 140, 256, true},
                          {333, 256, 256, true}, {5, 17, 3, false},  {1000, 140, 1, true}};
    std::mt19937 rng(611);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (const Case& c : cases) {
        std::vector<float> x(size_t(c.T) * c.K), w(size_t(c.N) * c.K), b(c.N);
        for (auto& v : x)
            v = u(rng);
        for (auto& v : w)
            v = u(rng) * 0.1f;
        for (auto& v : b)
            v = u(rng);
        std::vector<float> fast(size_t(c.T) * c.N, NAN), slow(size_t(c.T) * c.N, NAN);
        const float* bp = c.bias ? b.data() : nullptr;
        core_cpu_linear::run(x.data(), w.data(), bp, fast.data(), c.T, c.K, c.N, false);
        core_cpu_linear::run(x.data(), w.data(), bp, slow.data(), c.T, c.K, c.N, true);
        INFO("T=" << c.T << " K=" << c.K << " N=" << c.N << " bias=" << c.bias);
        for (float v : fast)
            REQUIRE(std::isfinite(v)); // every output written (0 beta, no stale NaN)
        REQUIRE(max_rel_err(fast, slow) < 1e-5);
    }
}

TEST_CASE("cpu_linear: scalar loop computes x @ w^T + b", "[unit][cpu_linear]") {
    // x[2,3], w[2,3] -> out[2,2]
    const float x[] = {1, 2, 3, 4, 5, 6};
    const float w[] = {1, 0, -1, 0.5f, 0.5f, 0.5f};
    const float b[] = {10, -1};
    float out[4];
    for (bool scalar : {true, false}) {
        core_cpu_linear::run(x, w, b, out, 2, 3, 2, scalar);
        CHECK(out[0] == 1 - 3 + 10.0f);
        CHECK(out[1] == 3 - 1.0f);
        CHECK(out[2] == 4 - 6 + 10.0f);
        CHECK(out[3] == 7.5f - 1.0f);
    }
}
