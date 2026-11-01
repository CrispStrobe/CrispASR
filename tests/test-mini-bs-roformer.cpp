// Model-free failure and artifact guards for Mini BS RoFormer V2.5.
#include <catch2/catch_test_macros.hpp>

#include "mini_bs_roformer.h"
#include "mini_bs_roformer_fft.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
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
