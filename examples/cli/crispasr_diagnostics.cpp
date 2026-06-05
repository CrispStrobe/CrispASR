// crispasr_diagnostics.cpp — see header for the why.

#include "crispasr_diagnostics.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// The CMake macros below are defined in examples/cli/CMakeLists.txt.
// They are wrapped in `defined(...) ? ... : "?"` rather than fall-through
// strings so that when the project is built outside the standard CMake flow
// (eg. someone copying cli.cpp into another tree) the binary still links.

#ifndef CRISPASR_VERSION_STR
#define CRISPASR_VERSION_STR "unknown"
#endif
#ifndef CRISPASR_GIT_SHA1
#define CRISPASR_GIT_SHA1 "unknown"
#endif
#ifndef CRISPASR_GIT_DATE
#define CRISPASR_GIT_DATE "unknown"
#endif
#ifndef CRISPASR_GIT_SUBJECT
#define CRISPASR_GIT_SUBJECT ""
#endif
#ifndef CRISPASR_BUILD_DATE
#define CRISPASR_BUILD_DATE __DATE__ " " __TIME__
#endif
#ifndef CRISPASR_BUILD_TYPE
#define CRISPASR_BUILD_TYPE "unknown"
#endif
#ifndef CRISPASR_CUDA_ARCHS
#define CRISPASR_CUDA_ARCHS ""
#endif
#ifndef CRISPASR_CUDA_VERSION
#define CRISPASR_CUDA_VERSION ""
#endif

// One macro per compiled-in ggml backend. Each guard mirrors the
// corresponding GGML_* CMake option so the banner accurately reports
// which paths the binary can drive at runtime.
static const char* crispasr_compiled_backends() {
    static std::string s;
    if (!s.empty())
        return s.c_str();
    s = "cpu";
#ifdef CRISPASR_HAVE_CUDA
    s += ",cuda";
#endif
#ifdef CRISPASR_HAVE_METAL
    s += ",metal";
#endif
#ifdef CRISPASR_HAVE_VULKAN
    s += ",vulkan";
#endif
#ifdef CRISPASR_HAVE_HIP
    s += ",hip";
#endif
#ifdef CRISPASR_HAVE_SYCL
    s += ",sycl";
#endif
#ifdef CRISPASR_HAVE_OPENCL
    s += ",opencl";
#endif
#ifdef CRISPASR_HAVE_BLAS
    s += ",blas";
#endif
#ifdef CRISPASR_HAVE_MUSA
    s += ",musa";
#endif
    return s.c_str();
}

static const char* crispasr_compiler_id() {
#if defined(__clang__)
    static char buf[64];
    std::snprintf(buf, sizeof(buf), "clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
    return buf;
#elif defined(__GNUC__)
    static char buf[64];
    std::snprintf(buf, sizeof(buf), "gcc %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    return buf;
#elif defined(_MSC_VER)
    static char buf[64];
    std::snprintf(buf, sizeof(buf), "msvc %d", _MSC_VER);
    return buf;
#else
    return "unknown";
#endif
}

void crispasr_print_short_banner(FILE* out) {
    std::fprintf(out, "crispasr %s (git %s, %s) [backends: %s]\n", CRISPASR_VERSION_STR, CRISPASR_GIT_SHA1,
                 CRISPASR_BUILD_TYPE, crispasr_compiled_backends());
}

void crispasr_print_build_info(FILE* out) {
    std::fprintf(out, "=== build info ===\n");
    std::fprintf(out, "  version       : %s\n", CRISPASR_VERSION_STR);
    std::fprintf(out, "  git sha       : %s\n", CRISPASR_GIT_SHA1);
    if (CRISPASR_GIT_DATE[0] != '\0' && std::strcmp(CRISPASR_GIT_DATE, "unknown") != 0) {
        std::fprintf(out, "  git date      : %s\n", CRISPASR_GIT_DATE);
    }
    if (CRISPASR_GIT_SUBJECT[0] != '\0') {
        std::fprintf(out, "  git subject   : %s\n", CRISPASR_GIT_SUBJECT);
    }
    std::fprintf(out, "  build date    : %s\n", CRISPASR_BUILD_DATE);
    std::fprintf(out, "  build type    : %s\n", CRISPASR_BUILD_TYPE);
    std::fprintf(out, "  compiler      : %s\n", crispasr_compiler_id());
#if defined(__APPLE__)
    std::fprintf(out, "  os            : darwin\n");
#elif defined(__linux__)
    std::fprintf(out, "  os            : linux\n");
#elif defined(_WIN32)
    std::fprintf(out, "  os            : windows\n");
#endif
#if defined(__x86_64__) || defined(_M_X64)
    std::fprintf(out, "  arch          : x86_64\n");
#elif defined(__aarch64__) || defined(_M_ARM64)
    std::fprintf(out, "  arch          : aarch64\n");
#endif
    std::fprintf(out, "  ggml backends : %s\n", crispasr_compiled_backends());
#ifdef CRISPASR_HAVE_CUDA
    if (CRISPASR_CUDA_VERSION[0] != '\0') {
        std::fprintf(out, "  cuda toolkit  : %s\n", CRISPASR_CUDA_VERSION);
    }
    if (CRISPASR_CUDA_ARCHS[0] != '\0') {
        std::fprintf(out, "  cuda archs    : %s\n", CRISPASR_CUDA_ARCHS);
    }
#endif
    std::fprintf(out, "\n");
}

static void print_env_if_set(FILE* out, const char* name) {
    const char* v = std::getenv(name);
    if (v && *v) {
        std::fprintf(out, "  %-28s = %s\n", name, v);
    } else {
        std::fprintf(out, "  %-28s = (unset)\n", name);
    }
}

#if defined(__linux__)
// Best-effort: dump small text files that are likely to point at the cause
// of a CUDA bring-up failure. Silent when the file does not exist.
static void cat_small_file(FILE* out, const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f)
        return;
    std::fprintf(out, "  --- %s ---\n", path);
    char buf[512];
    int line_count = 0;
    while (std::fgets(buf, sizeof(buf), f) && line_count < 20) {
        std::fprintf(out, "  %s", buf);
        line_count++;
    }
    if (line_count == 20 && !std::feof(f)) {
        std::fprintf(out, "  ... (truncated)\n");
    }
    std::fclose(f);
}

#include <dirent.h>
#include <sys/stat.h>

static void list_dir_files(FILE* out, const char* path, const char* substr) {
    DIR* d = ::opendir(path);
    if (!d)
        return;
    std::fprintf(out, "  %s/\n", path);
    struct dirent* e;
    int n = 0;
    while ((e = ::readdir(d)) != nullptr) {
        if (e->d_name[0] == '.')
            continue;
        if (substr && !std::strstr(e->d_name, substr))
            continue;
        std::fprintf(out, "    %s\n", e->d_name);
        n++;
        if (n > 40) {
            std::fprintf(out, "    ... (truncated)\n");
            break;
        }
    }
    ::closedir(d);
}
#endif

void crispasr_print_runtime_env(FILE* out) {
    std::fprintf(out, "=== runtime env ===\n");
    print_env_if_set(out, "PATH");
    print_env_if_set(out, "LD_LIBRARY_PATH");
    print_env_if_set(out, "LIBRARY_PATH");
    print_env_if_set(out, "CUDA_HOME");
    print_env_if_set(out, "CUDA_PATH");
    print_env_if_set(out, "CUDA_VISIBLE_DEVICES");
    print_env_if_set(out, "NVIDIA_VISIBLE_DEVICES");
    print_env_if_set(out, "NVIDIA_DRIVER_CAPABILITIES");
    print_env_if_set(out, "GGML_CUDA_DEBUG");
    print_env_if_set(out, "GGML_VK_VISIBLE_DEVICES");
    print_env_if_set(out, "GGML_VK_PIPELINE_CACHE_DEBUG");
    print_env_if_set(out, "CRISPASR_USE_CUDA_COMPAT");
    print_env_if_set(out, "CRISPASR_BACKEND");
    print_env_if_set(out, "CRISPASR_CACHE_DIR");

#if defined(__linux__)
    std::fprintf(out, "\n=== ld.so.conf.d (cuda-compat shadows host libcuda when present) ===\n");
    list_dir_files(out, "/etc/ld.so.conf.d", "cuda");
    cat_small_file(out, "/etc/ld.so.conf.d/000_cuda_compat.conf");
    cat_small_file(out, "/etc/ld.so.conf.d/cuda-compat.conf");

    std::fprintf(out, "\n=== /usr/local/cuda/compat (forward-compat libcuda — opt-in only) ===\n");
    list_dir_files(out, "/usr/local/cuda/compat", nullptr);

    std::fprintf(out, "\n=== nvidia driver (host) ===\n");
    cat_small_file(out, "/proc/driver/nvidia/version");
#endif
    std::fprintf(out, "\n");
}

void crispasr_print_devices(FILE* out) {
    std::fprintf(out, "=== ggml backends + devices ===\n");

    // ggml_backend_load_all() walks the build's loadable backends and
    // registers their device classes. On CUDA this calls cudaGetDeviceCount,
    // which fails gracefully (and logs through GGML_LOG_ERROR) when the
    // host driver is incompatible with the runtime — we don't crash on
    // that path, the registry just ends up CPU-only.
    ggml_backend_load_all();

    const size_t n_reg = ggml_backend_reg_count();
    std::fprintf(out, "  registered backends: %zu\n", n_reg);
    for (size_t i = 0; i < n_reg; ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        const char* name = ggml_backend_reg_name(reg);
        const size_t n_dev = ggml_backend_reg_dev_count(reg);
        std::fprintf(out, "    [%zu] %s (devices: %zu)\n", i, name ? name : "(null)", n_dev);
    }

    const size_t n_dev = ggml_backend_dev_count();
    std::fprintf(out, "  registered devices : %zu\n", n_dev);
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        struct ggml_backend_dev_props p {};
        ggml_backend_dev_get_props(dev, &p);
        const char* type_str = "?";
        switch (p.type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            type_str = "cpu";
            break;
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            type_str = "gpu";
            break;
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            type_str = "igpu";
            break;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            type_str = "accel";
            break;
        case GGML_BACKEND_DEVICE_TYPE_META:
            type_str = "meta";
            break;
        }
        std::fprintf(out, "    [%zu] %-6s name=%s desc=%s mem=%zu/%zu MiB id=%s\n", i, type_str, p.name ? p.name : "?",
                     p.description ? p.description : "?", (size_t)(p.memory_free / (1024 * 1024)),
                     (size_t)(p.memory_total / (1024 * 1024)), p.device_id ? p.device_id : "?");
    }
    std::fprintf(out, "\n");
}

void crispasr_print_full_diagnostics(FILE* out) {
    crispasr_print_build_info(out);
    crispasr_print_runtime_env(out);
    crispasr_print_devices(out);
}
