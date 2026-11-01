// Pull the production backend's single translation unit into a minimal shared
// validation library. Paired CPU/GPU benchmarks can load its existing C API
// without building every unrelated backend in libcrispasr.
#include "mini_bs_roformer.h"

#if defined(_WIN32)
#define MINI_BSR_PROBE_EXPORT __declspec(dllexport)
#else
#define MINI_BSR_PROBE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" MINI_BSR_PROBE_EXPORT int mini_bs_roformer_validation_abi_version();
extern "C" MINI_BSR_PROBE_EXPORT int mini_bs_roformer_validation_abi_version() {
    // The reference keeps the static archive member live. The result is not a
    // claim about model availability or numerical/performance acceptance.
    (void)mini_bs_roformer_default_params();
    return 1;
}
