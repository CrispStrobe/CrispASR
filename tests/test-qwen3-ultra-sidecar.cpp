#include "../src/core/qwen3_ultra_sidecar.h"
#include "../src/core/ggml_cpu_backend.h"
#include <cstdio>

int main(int argc, char** argv) {
    qwen3_ultra_sidecar::Package package;
    package.mean[0] = 42.0f;
    if (qwen3_ultra_sidecar::load(nullptr, nullptr, package) || package.mean[0] != 42.0f)
        return 1;
    auto* backend = core_cpu_backend::init();
    if (!backend)
        return 2;
    if (qwen3_ultra_sidecar::load("", backend, package) || package.mean[0] != 42.0f) {
        ggml_backend_free(backend);
        return 3;
    }
    if (argc >= 3) {
        if (!qwen3_ultra_sidecar::load(argv[1], backend, package) || !package.weights.valid() ||
            package.mean[0] != 0.0f || package.stddev[0] != 1.0f || package.weights.audio0_weight[0] != 0.25f) {
            ggml_backend_free(backend);
            return 4;
        }
        package.mean[0] = 42.0f;
        for (int i = 2; i < argc; ++i) {
            if (qwen3_ultra_sidecar::load(argv[i], backend, package) || package.mean[0] != 42.0f ||
                package.weights.audio0_weight[0] != 0.25f) {
                ggml_backend_free(backend);
                return 5;
            }
        }
    } else if (argc != 1) {
        ggml_backend_free(backend);
        return 6;
    }
    ggml_backend_free(backend);
    std::puts("Ultra sidecar loader controls passed");
    return 0;
}
