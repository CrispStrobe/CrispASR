// crispasr_gpu_plugin_hint — how to fix a GPU backend plugin that failed to
// load, from the loader's own error text (#606). Header-only so the wording is
// unit-tested (tests/test-gpu-plugin-hint.cpp) without the CLI.
#pragma once

#include <cstring>
#include <string>

#if defined(_WIN32)
constexpr bool k_crispasr_hint_windows = true;
#else
constexpr bool k_crispasr_hint_windows = false;
#endif

// stem: "cuda", "hip", "vulkan", ...; err: dlerror() / LoadLibrary text.
// windows: give Windows advice (DLLs, zips) instead of Linux apt steps.
// Returns "" when there is no specific advice.
inline std::string crispasr_gpu_plugin_hint(const char* stem, const std::string& err,
                                            bool windows = k_crispasr_hint_windows) {
    const bool cuda = std::strcmp(stem, "cuda") == 0;
    const bool hip = std::strcmp(stem, "hip") == 0;
    std::string h;
    if (cuda && err.find("libcuda.so") != std::string::npos)
        return "the NVIDIA driver library is not visible (in a container: run with --gpus all / the NVIDIA runtime)";
    if (cuda && (err.find("cudart") != std::string::npos || err.find("cublas") != std::string::npos)) {
        const bool v13 = err.find(".so.13") != std::string::npos || err.find("64_13") != std::string::npos;
        const bool v12 = err.find(".so.12") != std::string::npos || err.find("64_12") != std::string::npos;
        const char* major = v13 ? "13" : v12 ? "12" : "";
        h = std::string("this package needs the CUDA ") + (major[0] ? major : "") + " runtime libraries (" +
            (windows ? std::string("cudart64 + cublas64 + cublasLt64") + (major[0] ? std::string(" _") + major : "") +
                           " DLLs"
                     : std::string("libcudart + libcublas") + (major[0] ? std::string(" .so.") + major : "")) +
            ").";
        if (windows) {
            h += " Use the self-contained crispasr-windows-x86_64-cuda zip (DLLs included), or put the CUDA "
                 "runtime DLLs of that major next to crispasr.exe. Steps: docs/troubleshooting.md (Windows: which "
                 "download, and the CUDA DLLs)";
            return h;
        }
        h += " The NVIDIA driver alone does not ship them, and the distro 'nvidia-cuda-toolkit' package is usually "
             "an older CUDA; the 'CUDA Version' in nvidia-smi is only what the driver supports. Add NVIDIA's CUDA "
             "apt repository (the cuda-keyring package) and install the two libraries";
        if (major[0])
            h += std::string(", e.g. `sudo apt install cuda-cudart-") + major + "-0 libcublas-" + major + "-0` (any " +
                 major + ".x works)";
        h += ", or use the CrispASR CUDA package that matches the CUDA major you have. Steps: docs/install.md "
             "(Prebuilt Linux tarballs)";
        return h;
    }
    if (hip && err.find("lib") != std::string::npos)
        return "this package needs the ROCm runtime with hipBLAS/rocBLAS of the matching major version "
               "(ROCm 6: crispasr-linux-x86_64-hip, ROCm 7: crispasr-linux-x86_64-hip7)";
    return "";
}
