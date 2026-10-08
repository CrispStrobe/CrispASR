// Apple-only lifecycle smoke test for the persistent MTLBinaryArchive cache.
// No compute graph runs here, so these checks do not prove PSO serialization.
#include <catch2/catch_test_macros.hpp>

#if __APPLE__
#import <Metal/Metal.h>

#include "ggml-metal-device.h"
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

struct MetalCacheFixture {
    std::optional<std::string> old_cache;
    std::optional<std::string> old_disable;
    fs::path tmp;

    MetalCacheFixture() {
        if (const char* v = std::getenv("GGML_METAL_PIPELINE_CACHE"))
            old_cache = v;
        if (const char* v = std::getenv("GGML_METAL_PIPELINE_CACHE_DISABLE"))
            old_disable = v;
        tmp = fs::temp_directory_path() / ("crispasr-test-metalcache-" + std::to_string(getpid()));
        fs::create_directories(tmp);
        setenv("GGML_METAL_PIPELINE_CACHE", tmp.c_str(), 1);
        unsetenv("GGML_METAL_PIPELINE_CACHE_DISABLE");
    }

    ~MetalCacheFixture() {
        restore("GGML_METAL_PIPELINE_CACHE", old_cache);
        restore("GGML_METAL_PIPELINE_CACHE_DISABLE", old_disable);
        std::error_code ec;
        fs::remove_all(tmp, ec);
    }

    static void restore(const char* name, const std::optional<std::string>& value) {
        if (value)
            setenv(name, value->c_str(), 1);
        else
            unsetenv(name);
    }
};

static bool has_metal_device() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        const bool available = device != nil;
        [device release];
        return available;
    }
}

TEST_CASE_METHOD(MetalCacheFixture, "metal pipeline cache: init + free + reinit is crash-free",
                 "[unit][metal][pipeline-cache]") {
    if (!has_metal_device())
        SKIP("No Metal device: cache lifecycle was not executed.");
    // Use separate internal device lifetimes so each free flushes its archive.
    // The second argument is the number of virtual devices (one here).
    auto* dev = ggml_metal_device_init(0, 1);
    REQUIRE(dev != nullptr);
    ggml_metal_device_free(dev);
    auto* dev2 = ggml_metal_device_init(0, 1);
    REQUIRE(dev2 != nullptr);
    ggml_metal_device_free(dev2);
}

TEST_CASE_METHOD(MetalCacheFixture, "metal pipeline cache: DISABLE env var skips archive creation",
                 "[unit][metal][pipeline-cache]") {
    if (!has_metal_device())
        SKIP("No Metal device: cache disable behavior was not executed.");
    setenv("GGML_METAL_PIPELINE_CACHE_DISABLE", "1", 1);
    auto* dev = ggml_metal_device_init(0, 1);
    REQUIRE(dev != nullptr);
    ggml_metal_device_free(dev);
    for (const auto& entry : fs::directory_iterator(tmp))
        REQUIRE(entry.path().extension() != ".archive");
}

#else
TEST_CASE("metal pipeline cache: skipped on non-Apple", "[unit][metal][pipeline-cache]") {
    SKIP("Metal backend not compiled on this platform.");
}
#endif
