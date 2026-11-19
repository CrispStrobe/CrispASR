// Backend device selection shared by the model loaders.
#pragma once

#include "ggml-backend.h"

// --gpu picks a discrete GPU, else an integrated one, else the CPU. ggml reports
// unified-memory GPUs (AMD APUs and Intel iGPUs on Vulkan, for example) as IGPU
// rather than GPU, so asking for GPU alone silently falls back to the CPU there.
static inline ggml_backend_dev_t zonos3_backend_device(bool use_gpu) {
    ggml_backend_dev_t dev = nullptr;
    if (use_gpu)
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (use_gpu && !dev)
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!dev)
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return dev;
}

static inline bool zonos3_backend_is_gpu(ggml_backend_dev_t dev) {
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}
