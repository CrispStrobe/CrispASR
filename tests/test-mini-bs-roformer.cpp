// Model-free failure and artifact guards for Mini BS RoFormer V2.5.
#include <catch2/catch_test_macros.hpp>

#include "core/ggml_cpu_backend.h"
#include "mini_bs_roformer.h"
#include "mini_bs_roformer_bf16.h"
#include "mini_bs_roformer_fft.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {
struct MetadataFile {
    std::string path;
    explicit MetadataFile(const char* arch) {
        path = (std::filesystem::temp_directory_path() /
                ("mini-bs-roformer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 ".gguf"))
                   .string();
        gguf_context* meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", arch);
        const bool ok = gguf_write_to_file(meta, path.c_str(), true);
        gguf_free(meta);
        REQUIRE(ok);
    }
    ~MetadataFile() { std::remove(path.c_str()); }
};
} // namespace

TEST_CASE("Mini BS RoFormer rejects unavailable and incomplete models", "[unit][mini-bs-roformer]") {
    auto params = mini_bs_roformer_default_params();
    params.use_gpu = false;
    params.n_threads = 1;
    SECTION("missing file") {
        CHECK(mini_bs_roformer_init_from_file("/no-such-mini-bs-roformer-model.gguf", params) == nullptr);
    }
    SECTION("different architecture") {
        MetadataFile file("mel-band-roformer");
        CHECK(mini_bs_roformer_init_from_file(file.path.c_str(), params) == nullptr);
    }
    SECTION("known architecture with no tensors") {
        MetadataFile file("mini-bs-roformer");
        CHECK(mini_bs_roformer_init_from_file(file.path.c_str(), params) == nullptr);
    }
}

TEST_CASE("Mini BS RoFormer rejects separation without a context", "[unit][mini-bs-roformer]") {
    float stereo[2] = {0.0f, 0.0f};
    CHECK(mini_bs_roformer_separate(nullptr, stereo, 1, 2) == nullptr);
    CHECK(mini_bs_roformer_separate(nullptr, nullptr, 0, 2) == nullptr);
    CHECK(mini_bs_roformer_backend_name(nullptr) == nullptr);
    CHECK(mini_bs_roformer_compute_precision(nullptr) == nullptr);
    mini_bs_roformer_result_free(nullptr);
    mini_bs_roformer_free(nullptr);
}

TEST_CASE("Mini BS RoFormer FFT preserves quiet-band phase accuracy", "[unit][mini-bs-roformer]") {
    const double pi = std::acos(-1.0);
    for (int size : {2048, 4096}) {
        INFO("FFT size " << size);
        mini_bs_fft::Plan plan(size);
        std::vector<float> real(size, 0), imag(size, 0);
        const int impulse = 137;
        real[impulse] = 1;
        plan.transform(real.data(), imag.data());
        double max_error = 0;
        for (int k = 0; k < size; ++k) {
            const double phase = -2.0 * pi * k * impulse / size;
            max_error = std::max(max_error, std::abs(real[k] - std::cos(phase)));
            max_error = std::max(max_error, std::abs(imag[k] - std::sin(phase)));
            imag[k] = -imag[k];
        }
        // The former recursive F32 twiddle update drifts by about 3e-5
        // here. An analytical shifted impulse provides an independent oracle.
        CHECK(max_error < 1e-6);
        plan.transform(real.data(), imag.data());
        double inverse_error = 0;
        for (int k = 0; k < size; ++k) {
            inverse_error = std::max(inverse_error, std::abs((double)real[k] / size - (k == impulse ? 1 : 0)));
            inverse_error = std::max(inverse_error, std::abs((double)imag[k] / size));
        }
        CHECK(inverse_error < 1e-6);
    }
    CHECK_THROWS_AS(mini_bs_fft::Plan(0), std::invalid_argument);
    CHECK_THROWS_AS(mini_bs_fft::Plan(2), std::invalid_argument);
    CHECK_THROWS_AS(mini_bs_fft::Plan(4095), std::invalid_argument);
    CHECK_THROWS_AS(mini_bs_fft::Plan(8192), std::invalid_argument);
}

TEST_CASE("Mini BS RoFormer BF16 packed graph matches independent dot products", "[unit][mini-bs-roformer]") {
    CHECK_FALSE(mini_bs_bf16::supported(nullptr));
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(core_cpu_backend::init(), ggml_backend_free);
    REQUIRE(cpu != nullptr);
    // Exercise both small-M dispatches, a final 16-feature panel, and tails
    // owned by multiple workers. M205 also spans three 96-row blocks, with
    // a partial final block and an eight-row microkernel tail.
    for (const auto shape : {std::array<int, 2>{18, 13}, {514, 13}, {514, 205}}) {
        constexpr int n = 80;
        const int k = shape[0], m = shape[1];
        INFO("K=" << k << ", M=" << m);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2 * 1024 * 1024, nullptr, false}),
                                                                ggml_free);
        REQUIRE(ctx != nullptr);
        auto* w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, n);
        auto* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, m);
        CHECK(mini_bs_bf16::pack(ctx.get(), ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, 17, n)) == nullptr);
        CHECK(mini_bs_bf16::pack(ctx.get(), ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, 30)) == nullptr);
        CHECK(mini_bs_bf16::pack(ctx.get(), ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, n)) == nullptr);
        auto* packed = mini_bs_bf16::pack(ctx.get(), w);
        if (!mini_bs_bf16::hardware_supported()) {
            CHECK_FALSE(mini_bs_bf16::supported(cpu.get()));
            CHECK(packed == nullptr);
            CHECK(mini_bs_bf16::graph(ctx.get(), packed, x, n) == nullptr);
            continue;
        }
        REQUIRE(mini_bs_bf16::supported(cpu.get()));
        REQUIRE(packed);
        CHECK(mini_bs_bf16::graph(ctx.get(), packed, x, n + 16) == nullptr);
        CHECK(mini_bs_bf16::graph(ctx.get(), packed, ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m), n) == nullptr);
        auto* output = mini_bs_bf16::graph(ctx.get(), packed, x, n);
        REQUIRE(output);
        auto* graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(graph, output);
        auto* weights = static_cast<ggml_bf16_t*>(w->data);
        auto* values = static_cast<ggml_bf16_t*>(x->data);
        for (int col = 0; col < n; ++col)
            for (int j = 0; j < k; ++j)
                weights[col * k + j] = ggml_fp32_to_bf16(((col * 17 + j * 5) % 37 - 18) / 31.0f + col * 0.003f);
        for (int row = 0; row < m; ++row)
            for (int j = 0; j < k; ++j)
                values[row * k + j] = ggml_fp32_to_bf16(((row * 7 + j * 11) % 41 - 20) / 29.0f - row * 0.009f);
        auto* actual = static_cast<float*>(output->data);
        std::fill(actual, actual + n * m, std::numeric_limits<float>::quiet_NaN());
        REQUIRE(core_cpu_backend::compute(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS);
        std::vector<float> one_worker(actual, actual + n * m);
        std::fill(actual, actual + n * m, std::numeric_limits<float>::quiet_NaN());
        REQUIRE(core_cpu_backend::compute(ctx.get(), graph, 2) == GGML_STATUS_SUCCESS);
        CHECK(std::memcmp(actual, one_worker.data(), one_worker.size() * sizeof(float)) == 0);
        for (int row = 0; row < m; ++row)
            for (int col = 0; col < n; ++col) {
                double expected = 0;
                for (int j = 0; j < k; ++j)
                    expected +=
                        (double)ggml_bf16_to_fp32(weights[col * k + j]) * ggml_bf16_to_fp32(values[row * k + j]);
                CHECK(std::isfinite(actual[row * n + col]));
                CHECK(std::abs(actual[row * n + col] - expected) <= 2e-5 * (1 + std::abs(expected)));
            }
    }
}
