#include "crispasr_imatrix.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

namespace {
void environment(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value)
        setenv(key, value, 1);
    else
        unsetenv(key);
#endif
}
} // namespace

TEST_CASE("external scheduler callback collects real activation columns", "[unit][imatrix]") {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("crispasr-imatrix-" + std::to_string(stamp) + ".gguf");
    environment("CRISPASR_IMATRIX_OUT", path.string().c_str());
    environment("CRISPASR_ACTDUMP_OUT", nullptr);
    auto callback = crispasr_imatrix_callback();
    REQUIRE(callback);
    const ggml_init_params params = {1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    REQUIRE(ctx);
    auto* weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_set_name(weight, "blk.0.ffn_up.weight");
    auto* activation = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    auto* product = ggml_mul_mat(ctx.get(), weight, activation);
    auto* unrelated = ggml_add(ctx.get(), activation, activation);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    REQUIRE(backend);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    REQUIRE(buffer);
    const float rows[] = {1, 2, 3, 4, 5, 6};
    ggml_backend_tensor_set(activation, rows, 0, sizeof(rows));
    CHECK_FALSE(callback(unrelated, true, nullptr));
    REQUIRE(callback(product, true, nullptr));
    REQUIRE(callback(product, false, nullptr));
    // Repeated prefill/decode computations accumulate rows; callback userdata
    // is intentionally NULL, separate from Index-Echo's stage-capture context.
    REQUIRE(callback(product, false, nullptr));
    crispasr_imatrix_flush();
    REQUIRE(std::filesystem::exists(path));
    ggml_context* loaded = nullptr;
    const gguf_init_params read_params = {false, &loaded};
    auto* file = gguf_init_from_file(path.string().c_str(), read_params);
    REQUIRE(file);
    REQUIRE(loaded);
    REQUIRE(gguf_get_n_tensors(file) == 1);
    const int64_t key = gguf_find_key(file, "count.blk.0.ffn_up.weight");
    REQUIRE(key >= 0);
    CHECK(gguf_get_val_u64(file, key) == 4);
    auto* tensor = ggml_get_tensor(loaded, "blk.0.ffn_up.weight");
    REQUIRE(tensor);
    REQUIRE(tensor->ne[0] == 3);
    const auto* sums = static_cast<const float*>(tensor->data);
    CHECK(sums[0] == 34);
    CHECK(sums[1] == 58);
    CHECK(sums[2] == 90);
    gguf_free(file);
    ggml_free(loaded);
    std::filesystem::remove(path);
}
