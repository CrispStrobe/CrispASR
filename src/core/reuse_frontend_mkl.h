#pragma once

// Optional CPU research profile. This header has no oneMKL dependency; only
// explicitly enabled research targets link the matching implementation.
#include "reuse_frontend.h"
#include <string>

namespace reuse_frontend_mkl {
enum class Window { HighAccuracy, LowAccuracy, EnhancedPerformance, Libm };
struct Analysis {
    std::vector<float> window, real, imaginary, magnitude, phase;
};
Analysis transform(const std::vector<float>& audio, Window mode = Window::HighAccuracy, bool batch = true);
std::string version();
// Independent CPU Hann generation, shared with the optional cuFFT frontend.
std::vector<float> periodic_window(Window mode = Window::HighAccuracy);
bool analyze(const std::vector<float>& audio, int sample_rate, reuse_frontend::Spectrum& output,
             reuse_frontend::Diagnostics* diagnostics = nullptr);
} // namespace reuse_frontend_mkl
