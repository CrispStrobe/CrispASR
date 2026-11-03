// Pull the production backend's single translation unit into a minimal shared
// validation library. Paired CPU/GPU benchmarks can load its existing C API
// without building every unrelated backend in libcrispasr.
#include "scnet.h"

#if defined(_WIN32)
#define SCNET_PROBE_EXPORT __declspec(dllexport)
#else
#define SCNET_PROBE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" SCNET_PROBE_EXPORT int scnet_validation_abi_version();
extern "C" SCNET_PROBE_EXPORT int scnet_validation_abi_version() {
    // The reference keeps the static archive member live. The result is not a
    // claim about model availability or numerical/performance acceptance.
    (void)scnet_default_params();
    return 1;
}
