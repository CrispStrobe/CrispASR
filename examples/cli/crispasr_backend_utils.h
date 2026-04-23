#pragma once

#include "whisper_params.h"

inline bool crispasr_backend_should_use_gpu(const whisper_params& params) {
    return params.use_gpu && params.gpu_backend != "cpu";
}
