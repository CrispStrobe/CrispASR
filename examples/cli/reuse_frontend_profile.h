#pragma once

#include "core/reuse_frontend.h"
#ifdef CRISPASR_REUSE_HAVE_MKL
#include "core/reuse_frontend_mkl.h"
#endif
#include <cstdio>
#include <cstring>

namespace reuse_frontend_profile {
using Analyze = decltype(&reuse_frontend::analyze);

inline bool select(const char* requested, Analyze& analyze, const char*& label) {
    analyze = nullptr;
    label = nullptr;
    if (!requested || !*requested || std::strcmp(requested, "native") == 0) {
        analyze = reuse_frontend::analyze;
        label = "native";
        return true;
    }
    if (std::strcmp(requested, "mkl") != 0) {
        std::fprintf(stderr, "reuse: unknown CRISPASR_REUSE_FRONTEND profile '%s'; choose native or mkl\n", requested);
        return false;
    }
#ifdef CRISPASR_REUSE_HAVE_MKL
    analyze = reuse_frontend_mkl::analyze;
    label = "mkl";
    return true;
#else
    std::fprintf(stderr, "reuse: mkl frontend unavailable; configure CRISPASR_REUSE_MKL_FRONTEND=ON with an external "
                         "oneMKL SDK (the default build does not link oneMKL)\n");
    return false;
#endif
}
} // namespace reuse_frontend_profile
