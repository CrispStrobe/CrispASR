#pragma once

#include "core/reuse_frontend.h"
#ifdef CRISPASR_REUSE_HAVE_MKL
#include "core/reuse_frontend_mkl.h"
#endif
#ifdef CRISPASR_REUSE_HAVE_CUDA_FRONTEND
#include "core/reuse_frontend_cuda.h"
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace reuse_frontend_profile {
using Analyze = decltype(&reuse_frontend::analyze);

// The ISA limit must be inherited at process startup. Changing oneMKL's global
// dispatch after another user initialized it is unsafe; this selector never does so.
inline bool avx2_startup() {
    const char* instructions = std::getenv("MKL_ENABLE_INSTRUCTIONS");
    if (!instructions || std::strcmp(instructions, "AVX2")) {
        std::fprintf(stderr, "reuse: mkl-avx2 requires you to start the process with MKL_ENABLE_INSTRUCTIONS=AVX2\n");
        return false;
    }
    const char* cbwr = std::getenv("MKL_CBWR");
    if (cbwr && *cbwr) {
        std::fprintf(stderr, "reuse: mkl-avx2 requires you to unset MKL_CBWR before process startup\n");
        return false;
    }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx2"))
        return true;
#endif
    std::fprintf(stderr,
                 "reuse: mkl-avx2 requires an x86 host with AVX2 and a compiler supporting runtime detection\n");
    return false;
}

inline bool select(const char* requested, Analyze& analyze, const char*& label) {
    analyze = nullptr;
    label = nullptr;
    if (!requested || !*requested || std::strcmp(requested, "native") == 0) {
        analyze = reuse_frontend::analyze;
        label = "native";
        return true;
    }
    if (std::strcmp(requested, "cufft") == 0) {
#ifdef CRISPASR_REUSE_HAVE_CUDA_FRONTEND
        std::string reason;
        if (!reuse_frontend_cuda::available(reason)) {
            std::fprintf(stderr, "reuse: cuFFT frontend unavailable: %s\n", reason.c_str());
            return false;
        }
        analyze = reuse_frontend_cuda::analyze;
        label = "cufft";
        return true;
#else
        std::fprintf(stderr, "reuse: cuFFT frontend unavailable; configure CRISPASR_REUSE_CUDA_FRONTEND=ON and "
                             "CRISPASR_REUSE_MKL_FRONTEND=ON with external CUDA/oneMKL SDKs\n");
        return false;
#endif
    }
    const bool avx2 = std::strcmp(requested, "mkl-avx2") == 0;
    if (std::strcmp(requested, "mkl") != 0 && !avx2) {
        std::fprintf(stderr,
                     "reuse: unknown CRISPASR_REUSE_FRONTEND profile '%s'; choose native, mkl, mkl-avx2 or cufft\n",
                     requested);
        return false;
    }
#ifdef CRISPASR_REUSE_HAVE_MKL
    if (avx2 && !avx2_startup())
        return false;
    analyze = reuse_frontend_mkl::analyze;
    label = avx2 ? "mkl-avx2" : "mkl";
    return true;
#else
    std::fprintf(stderr, "reuse: mkl frontend unavailable; configure CRISPASR_REUSE_MKL_FRONTEND=ON with an external "
                         "oneMKL SDK (the default build does not link oneMKL)\n");
    return false;
#endif
}
} // namespace reuse_frontend_profile
