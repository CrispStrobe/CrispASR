#pragma once

// Public host interface; CUDA headers/dependencies belong only to optional
// research targets. No model weights or reference arrays enter this frontend.
#include "reuse_frontend.h"
#include <string>

namespace reuse_frontend_cuda {
struct Analysis {
    std::vector<float> window, real, imaginary, magnitude, phase;
};
bool available(std::string& reason);
std::string version();
Analysis transform(const std::vector<float>& audio);
bool analyze(const std::vector<float>& audio, int sample_rate, reuse_frontend::Spectrum& output,
             reuse_frontend::Diagnostics* diagnostics = nullptr);
} // namespace reuse_frontend_cuda
