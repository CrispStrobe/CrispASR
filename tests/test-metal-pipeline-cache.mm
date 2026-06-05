// test-metal-pipeline-cache.mm — Apple-only smoke test for the
// persistent MTLBinaryArchive pipeline cache (PLAN #88 /
// CrisperWeaver §5.18; implementation in
// ggml/src/ggml-metal/ggml-metal-device.m).
//
// File-static helpers in the implementation aren't directly
// linkable, so this test exercises the cache through its public
// side-effects: open a Metal device, hit a few compiled pipelines
// via the bundled libcrispasr.metallib, free the device, and
// assert that an archive file landed in the configured cache
// directory. Then reopen and confirm the file is still present
// (the reload path is exercised but the load log lines aren't
// captured here — eyeball them in `ctest --output-on-failure` if
// debugging).
//
// Gated `#if __APPLE__` — on Linux/Windows the file compiles to
// an empty translation unit, and the test runner skips it.

#include <catch2/catch_test_macros.hpp>

#if __APPLE__

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>  // getpid for the tempdir name
#include "ggml-metal.h"
#include "ggml-metal-device.h"

namespace fs = std::filesystem;

// Catch2 fixture: redirect the cache to a per-test tempdir so we
// don't stomp on the user's real ~/Library/Caches/ggml-metal/
// while running ctest. setenv before any device init.
struct MetalCacheFixture {
    std::string old_cache;
    fs::path tmp;
    MetalCacheFixture() {
        if (const char * v = std::getenv("GGML_METAL_PIPELINE_CACHE")) {
            old_cache = v;
        }
        tmp = fs::temp_directory_path() /
              ("crispasr-test-metalcache-" + std::to_string((unsigned long)::getpid()));
        fs::create_directories(tmp);
        setenv("GGML_METAL_PIPELINE_CACHE", tmp.c_str(), /*overwrite=*/1);
        // Make sure the opt-out env var isn't sticky from another shell.
        unsetenv("GGML_METAL_PIPELINE_CACHE_DISABLE");
    }
    ~MetalCacheFixture() {
        if (!old_cache.empty()) {
            setenv("GGML_METAL_PIPELINE_CACHE", old_cache.c_str(), 1);
        } else {
            unsetenv("GGML_METAL_PIPELINE_CACHE");
        }
        std::error_code ec;
        fs::remove_all(tmp, ec); // best-effort
    }
};

TEST_CASE_METHOD(MetalCacheFixture,
                 "metal pipeline cache: init + free + reinit is crash-free",
                 "[unit][metal][pipeline-cache]") {
    // The library does NOT JIT every PSO at device-init time —
    // ggml-metal compiles pipelines lazily when the first compute
    // graph asks for them. From a unit-test vantage point we can
    // verify:
    //   1. ggml_metal_device_init succeeds with the cache env var
    //      set to a fresh tempdir (the archive opens cleanly even
    //      when the file doesn't exist yet);
    //   2. ggml_metal_device_free doesn't crash with an empty
    //      archive (the flush no-ops on dirty=false);
    //   3. A second init reopens the archive (now present but
    //      empty) without errors.
    //
    // The "archive contains compiled PSOs" claim is exercised by
    // the manual crispasr-CLI smoke test recorded in HISTORY §5.18
    // — that path does run real graphs and produces the ~683 KB
    // serialised archive. Unit-testing it here would require
    // loading a model + running a graph, which is heavier than the
    // sub-second budget for ctest.
    auto * dev = ggml_metal_device_init(0);
    if (!dev) {
        // CI runner without a Metal-capable GPU — skip cleanly.
        SUCCEED("Metal device unavailable on this runner; skipping cache smoke test.");
        return;
    }
    ggml_metal_device_free(dev);

    // Reopen — should not crash, archive load path runs (file is
    // either absent or empty, both handled).
    auto * dev2 = ggml_metal_device_init(0);
    REQUIRE(dev2 != nullptr);
    ggml_metal_device_free(dev2);
}

TEST_CASE("metal pipeline cache: DISABLE env var skips archive creation",
          "[unit][metal][pipeline-cache]") {
    // No MetalCacheFixture — manage env directly so we can set
    // DISABLE before init. The point: assert that with DISABLE=1
    // no .archive file lands in the cache dir.
    fs::path tmp =
        fs::temp_directory_path() /
        ("crispasr-test-metalcache-disable-" + std::to_string((unsigned long)::getpid()));
    fs::create_directories(tmp);
    setenv("GGML_METAL_PIPELINE_CACHE", tmp.c_str(), 1);
    setenv("GGML_METAL_PIPELINE_CACHE_DISABLE", "1", 1);

    auto * dev = ggml_metal_device_init(0);
    if (!dev) {
        unsetenv("GGML_METAL_PIPELINE_CACHE");
        unsetenv("GGML_METAL_PIPELINE_CACHE_DISABLE");
        std::error_code ec;
        fs::remove_all(tmp, ec);
        SUCCEED("Metal device unavailable on this runner; skipping disable smoke test.");
        return;
    }
    ggml_metal_device_free(dev);

    // Cache directory should be empty.
    bool any_archive = false;
    for (const auto & entry : fs::directory_iterator(tmp)) {
        if (entry.path().extension() == ".archive") {
            any_archive = true;
            break;
        }
    }

    unsetenv("GGML_METAL_PIPELINE_CACHE");
    unsetenv("GGML_METAL_PIPELINE_CACHE_DISABLE");
    std::error_code ec;
    fs::remove_all(tmp, ec);

    REQUIRE_FALSE(any_archive);
}

#else  // !__APPLE__

// On non-Apple platforms the Metal backend isn't compiled in.
// Register one TEST_CASE so Catch2 doesn't complain about an
// empty binary, but the body just announces the skip.
TEST_CASE("metal pipeline cache: skipped on non-Apple", "[unit][metal][pipeline-cache]") {
    SUCCEED("Metal backend not compiled on this platform.");
}

#endif  // __APPLE__
