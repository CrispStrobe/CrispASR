// #606: the advice printed when libggml-cuda.so / ggml-hip fails to load.
#include "crispasr_gpu_plugin_hint.h"

#include <catch2/catch_test_macros.hpp>

static bool has(const std::string& s, const char* sub) {
    return s.find(sub) != std::string::npos;
}

TEST_CASE("gpu plugin hint: missing CUDA 13 runtime names the repo, packages and docs", "[unit][diag]") {
    const std::string h = crispasr_gpu_plugin_hint(
        "cuda", "libcublas.so.13: cannot open shared object file: No such file or directory", false);
    CHECK(has(h, "CUDA 13 runtime"));
    CHECK(has(h, "cuda-keyring"));
    CHECK(has(h, "cuda-cudart-13-0 libcublas-13-0"));
    CHECK(has(h, "any 13.x works"));
    CHECK(has(h, "nvidia-smi"));
    CHECK(has(h, "docs/install.md"));
}

TEST_CASE("gpu plugin hint: CUDA 12 on Linux", "[unit][diag]") {
    CHECK(has(crispasr_gpu_plugin_hint("cuda", "libcudart.so.12: cannot open", false), "cuda-cudart-12-0"));
}

TEST_CASE("gpu plugin hint: Windows gets DLL advice, never apt", "[unit][diag]") {
    const std::string h = crispasr_gpu_plugin_hint("cuda", "cublas64_13.dll not found", true);
    CHECK(has(h, "CUDA 13 runtime"));
    CHECK(has(h, "cublasLt64"));
    CHECK(has(h, "crispasr-windows-x86_64-cuda"));
    CHECK_FALSE(has(h, "apt"));
    CHECK_FALSE(has(h, "nvidia-cuda-toolkit"));
}

TEST_CASE("gpu plugin hint: unknown major still gives generic advice", "[unit][diag]") {
    const std::string h = crispasr_gpu_plugin_hint("cuda", "libcudart.so: cannot open", false);
    CHECK(has(h, "cuda-keyring"));
    CHECK_FALSE(has(h, "sudo apt install cuda-cudart-"));
}

TEST_CASE("gpu plugin hint: driver, HIP and no-advice cases", "[unit][diag]") {
    CHECK(has(crispasr_gpu_plugin_hint("cuda", "libcuda.so.1: cannot open"), "--gpus all"));
    CHECK(has(crispasr_gpu_plugin_hint("hip", "libhipblas.so.2: cannot open"), "ROCm"));
    CHECK(crispasr_gpu_plugin_hint("vulkan", "libvulkan.so.1: cannot open").empty());
    CHECK(crispasr_gpu_plugin_hint("cuda", "undefined symbol: foo").empty());
}
