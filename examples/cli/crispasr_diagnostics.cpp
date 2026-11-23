// crispasr_diagnostics.cpp — see header for the why.

#include "crispasr_diagnostics.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <unistd.h>
#endif

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
#ifndef CRISPASR_CUDA_RUNTIME_MAJOR
#define CRISPASR_CUDA_RUNTIME_MAJOR ""
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
        // Concatenate the compile-time value into the format literal. Besides
        // avoiding a needless format argument, this leaves the complete line
        // in the executable's string table so release CI can verify Windows
        // CUDA packages on hosted runners that have no NVIDIA driver.
        std::fprintf(out, "  cuda toolkit  : " CRISPASR_CUDA_VERSION " (build/link target)\n");
    }
    if (CRISPASR_CUDA_RUNTIME_MAJOR[0] != '\0') {
        std::fprintf(out, "  cuda runtime ABI: " CRISPASR_CUDA_RUNTIME_MAJOR
                          " (matching major-version runtime libraries required)\n");
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

    // Detect mixed CUDA major-version configs in ld.so.conf.d (#152).
    // E.g. both cuda-12.conf and cuda-13.conf — the higher-numbered file
    // can shadow libraries from the one the binary was compiled against,
    // causing silent heap corruption and crashes.
    {
        DIR* d = ::opendir("/etc/ld.so.conf.d");
        if (d) {
            int cuda_majors_seen = 0; // bitmask-ish: track distinct major versions
            int first_major = 0, second_major = 0;
            struct dirent* e;
            while ((e = ::readdir(d)) != nullptr) {
                if (!std::strstr(e->d_name, "cuda"))
                    continue;
                // Extract major version from names like "988_cuda-12.conf", "cuda-13.conf"
                const char* p = e->d_name;
                while (*p) {
                    if (p[0] == 'c' && p[1] == 'u' && p[2] == 'd' && p[3] == 'a' && p[4] == '-') {
                        int maj = std::atoi(p + 5);
                        if (maj >= 10 && maj <= 99) {
                            if (first_major == 0) {
                                first_major = maj;
                                cuda_majors_seen = 1;
                            } else if (maj != first_major) {
                                second_major = maj;
                                cuda_majors_seen = 2;
                            }
                        }
                        break;
                    }
                    ++p;
                }
            }
            ::closedir(d);
            if (cuda_majors_seen >= 2) {
                std::fprintf(out,
                             "\n  *** WARNING: mixed CUDA versions detected (CUDA %d and CUDA %d) ***\n"
                             "  Multiple CUDA major-version ld.so.conf.d entries can cause library\n"
                             "  mismatches, heap corruption, and crashes (see issue #152).\n"
                             "  Fix: remove the stale cuda-*.conf entry, run `sudo ldconfig`, and rebuild.\n"
                             "  Alternatively, rebuild with -DGGML_STATIC=ON to statically link CUDA.\n\n",
                             first_major, second_major);
            }
        }
    }

    std::fprintf(out, "\n=== /usr/local/cuda/compat (forward-compat libcuda — opt-in only) ===\n");
    list_dir_files(out, "/usr/local/cuda/compat", nullptr);

    std::fprintf(out, "\n=== nvidia driver (host) ===\n");
    cat_small_file(out, "/proc/driver/nvidia/version");
#endif
    std::fprintf(out, "\n");
}

// ── #606: why did a GPU plugin not load? ─────────────────────────────────────
namespace {

struct gpu_plugin_kind {
    const char* stem;     // file stem after the lib prefix: ggml-<stem>
    const char* reg_name; // ggml_backend_reg_name() once registered
};

const gpu_plugin_kind k_gpu_plugins[] = {
    {"cuda", "CUDA"}, {"hip", "ROCm"},  {"vulkan", "Vulkan"}, {"sycl", "SYCL"},
    {"musa", "MUSA"}, {"cann", "CANN"}, {"opencl", "OpenCL"},
};

std::filesystem::path executable_dir() {
#if defined(_WIN32)
    std::vector<wchar_t> buf(32768);
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
    if (n == 0 || n >= buf.size())
        return {};
    return std::filesystem::path(std::wstring(buf.data(), n)).parent_path();
#elif defined(__linux__)
    std::vector<char> buf(4096);
    const ssize_t n = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if (n <= 0)
        return {};
    return std::filesystem::path(std::string(buf.data(), (size_t)n)).parent_path();
#else
    return {}; // macOS builds link Metal statically; no GPU plugins to explain
#endif
}

bool reg_present(const char* name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        const char* n = ggml_backend_reg_name(ggml_backend_reg_get(i));
        if (n && std::strcmp(n, name) == 0)
            return true;
    }
    return false;
}

// Try to load the plugin the way ggml does; return "" on success, else the
// loader's error text.
std::string try_load(const std::filesystem::path& p) {
#if defined(_WIN32)
    HMODULE h = LoadLibraryExW(p.wstring().c_str(), nullptr,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (h) {
        FreeLibrary(h);
        return "";
    }
    const DWORD err = GetLastError();
    char msg[512] = {0};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, err, 0, msg, sizeof(msg),
                   nullptr);
    std::string s = "Windows error " + std::to_string((unsigned long)err) + ": " + msg;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
        s.pop_back();
    if (err == ERROR_MOD_NOT_FOUND)
        s += " (a DLL it depends on is missing)";
    return s;
#elif defined(__linux__)
    void* h = dlopen(p.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h) {
        dlclose(h);
        return "";
    }
    const char* e = dlerror();
    return e ? e : "unknown dlopen error";
#else
    (void)p;
    return "";
#endif
}

std::string hint_for(const char* stem, const std::string& err) {
    const bool cuda = std::strcmp(stem, "cuda") == 0;
    const bool hip = std::strcmp(stem, "hip") == 0;
    std::string h;
    if (cuda && err.find("libcuda.so") != std::string::npos)
        return "the NVIDIA driver library is not visible (in a container: run with --gpus all / the NVIDIA runtime)";
    if (cuda && (err.find("cudart") != std::string::npos || err.find("cublas") != std::string::npos)) {
        const bool v13 = err.find(".so.13") != std::string::npos || err.find("64_13") != std::string::npos;
        const bool v12 = err.find(".so.12") != std::string::npos || err.find("64_12") != std::string::npos;
        const char* major = v13 ? "13" : v12 ? "12" : "";
        h = std::string("this package needs the CUDA ") + (major[0] ? major : "") +
            " runtime libraries (libcudart + libcublas" + (major[0] ? std::string(" .so.") + major : "") +
            "). The NVIDIA driver alone does not ship them, and the distro 'nvidia-cuda-toolkit' "
            "package is usually an older CUDA. Install them from NVIDIA's repository";
        if (major[0])
            h += std::string(", e.g. `sudo apt install cuda-cudart-") + major + "-0 libcublas-" + major + "-0`";
        h += ", or use the CrispASR CUDA package that matches the CUDA major you have";
        return h;
    }
    if (hip && err.find("lib") != std::string::npos)
        return "this package needs the ROCm runtime with hipBLAS/rocBLAS of the matching major version "
               "(ROCm 6: crispasr-linux-x86_64-hip, ROCm 7: crispasr-linux-x86_64-hip7)";
    return "";
}

} // namespace

int crispasr_report_unloaded_gpu_plugins(FILE* out, bool compact) {
    const std::filesystem::path dir = executable_dir();
    if (dir.empty())
        return 0;
    std::error_code ec;
    int failed = 0;
    bool header = false;
    for (const auto& kind : k_gpu_plugins) {
        if (reg_present(kind.reg_name))
            continue;
        const std::string base = std::string("ggml-") + kind.stem;
#if defined(_WIN32)
        const std::vector<std::string> names = {base + ".dll"};
#else
        const std::vector<std::string> names = {"lib" + base + ".so", base + ".so"};
#endif
        for (const auto& name : names) {
            const std::filesystem::path p = dir / name;
            if (!std::filesystem::exists(p, ec))
                continue;
            std::string err = try_load(p);
            if (!compact && !header) {
                std::fprintf(out, "=== GPU backend plugins that did not register ===\n");
                header = true;
            }
            ++failed;
            if (err.empty())
                err = "the library loads, but its backend registered no device (driver too old for this "
                      "runtime, no supported GPU, or the device is hidden by *_VISIBLE_DEVICES)";
            const std::string hint = hint_for(kind.stem, err);
            if (compact) {
                std::fprintf(out, "crispasr: warning: %s backend not available, running on CPU: %s\n", kind.reg_name,
                             err.c_str());
                if (!hint.empty())
                    std::fprintf(out, "crispasr:          %s\n", hint.c_str());
            } else {
                std::fprintf(out, "  %s: %s\n", p.string().c_str(), err.c_str());
                if (!hint.empty())
                    std::fprintf(out, "    hint: %s\n", hint.c_str());
            }
            break;
        }
    }
    if (header)
        std::fprintf(out, "\n");
    return failed;
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
        default:
            type_str = "unknown";
            break;
        }
        std::fprintf(out, "    [%zu] %-6s name=%s desc=%s mem=%zu/%zu MiB id=%s\n", i, type_str, p.name ? p.name : "?",
                     p.description ? p.description : "?", (size_t)(p.memory_free / (1024 * 1024)),
                     (size_t)(p.memory_total / (1024 * 1024)), p.device_id ? p.device_id : "?");
    }
    std::fprintf(out, "\n");
    crispasr_report_unloaded_gpu_plugins(out, /*compact=*/false);
}

void crispasr_print_full_diagnostics(FILE* out) {
    crispasr_print_build_info(out);
    crispasr_print_runtime_env(out);
    crispasr_print_devices(out);
}
