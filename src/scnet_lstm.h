// Native recurrent kernel used as a scheduled ggml graph operation by SCNet.
// The complete input projection and output projection remain ordinary ggml
// matrix operations. This fuses the sequential hidden projection and four gates
// so a 20-second chunk does not create tens of thousands of tiny graph nodes.
#pragma once

#include "scnet_activation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(HAVE_BLAS)
#include <cblas.h>
#endif

namespace scnet_detail {

struct LstmScratch {
    std::vector<float> hidden, cell, gates;
    LstmScratch(int batch, int channels)
        : hidden(static_cast<size_t>(batch) * channels), cell(hidden.size()), gates(hidden.size() * 4) {}
};

struct LstmTiming {
    double gemm_ms = 0, gates_ms = 0, reset_ms = 0;
    uint64_t gemm_calls = 0, gate_steps = 0;
};

// All arrays use ordinary contiguous F32 storage. projected is [T,B,4H],
// recurrent_weight [4H,H], bias [4H], output [T,B,2H] (one direction written).
// Workers own disjoint batch rows; there are no cross-worker reductions.
// The ordinary specialization contains no clock reads or timing branches in
// either inner loop. Profiling is selected once per direction, by the caller.
template <bool Profile>
inline void lstm_direction_impl(const float* projected, const float* recurrent_weight, const float* recurrent_bias,
                                float* output, int hidden, int batch, int steps, bool reverse, int begin, int end,
                                bool use_blas, LstmScratch& scratch, LstmTiming* timing) {
    const auto gate_kernel = scnet_activation::selected_kernel();
    using Clock = std::chrono::steady_clock;
    Clock::time_point phase_start;
    if constexpr (Profile)
        phase_start = Clock::now();
    const int rows = end - begin, gates = 4 * hidden;
    std::fill(scratch.hidden.begin(), scratch.hidden.end(), 0);
    std::fill(scratch.cell.begin(), scratch.cell.end(), 0);
    if constexpr (Profile)
        timing->reset_ms += std::chrono::duration<double, std::milli>(Clock::now() - phase_start).count();
    for (int step = 0; step < steps; ++step) {
        const int time = reverse ? steps - step - 1 : step;
        if constexpr (Profile)
            phase_start = Clock::now();
        if (step == 0) {
            std::fill_n(scratch.gates.data(), static_cast<size_t>(rows) * gates, 0);
        } else {
#if defined(HAVE_ACCELERATE) || defined(HAVE_BLAS)
            if (use_blas) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, gates, hidden, 1.0f, scratch.hidden.data(),
                            hidden, recurrent_weight, hidden, 0.0f, scratch.gates.data(), gates);
            } else
#else
            (void)use_blas;
#endif
            {
                for (int row = 0; row < rows; ++row)
                    for (int gate = 0; gate < gates; ++gate) {
                        const float* h = scratch.hidden.data() + static_cast<size_t>(row) * hidden;
                        const float* w = recurrent_weight + static_cast<size_t>(gate) * hidden;
                        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        int k = 0;
                        for (; k + 3 < hidden; k += 4) {
                            a0 += h[k] * w[k];
                            a1 += h[k + 1] * w[k + 1];
                            a2 += h[k + 2] * w[k + 2];
                            a3 += h[k + 3] * w[k + 3];
                        }
                        float sum = (a0 + a1) + (a2 + a3);
                        for (; k < hidden; ++k)
                            sum += h[k] * w[k];
                        scratch.gates[static_cast<size_t>(row) * gates + gate] = sum;
                    }
            }
        }
        if constexpr (Profile) {
            const auto finished = Clock::now();
            const double milliseconds = std::chrono::duration<double, std::milli>(finished - phase_start).count();
            if (step == 0) {
                timing->reset_ms += milliseconds;
            } else {
                timing->gemm_ms += milliseconds;
                ++timing->gemm_calls;
            }
            phase_start = finished;
        }
        for (int row = 0; row < rows; ++row) {
            const float* px = projected + (static_cast<size_t>(time) * batch + begin + row) * gates;
            float* ph = scratch.gates.data() + static_cast<size_t>(row) * gates;
            float* h = scratch.hidden.data() + static_cast<size_t>(row) * hidden;
            float* c = scratch.cell.data() + static_cast<size_t>(row) * hidden;
            float* out =
                output + (static_cast<size_t>(time) * batch + begin + row) * 2 * hidden + (reverse ? hidden : 0);
            for (int k = 0; k < gates; ++k)
                ph[k] = (ph[k] + recurrent_bias[k]) + px[k];
            gate_kernel.update(ph, c, h, out, hidden);
        }
        if constexpr (Profile) {
            timing->gates_ms += std::chrono::duration<double, std::milli>(Clock::now() - phase_start).count();
            ++timing->gate_steps;
        }
    }
}

inline void lstm_direction(const float* projected, const float* recurrent_weight, const float* recurrent_bias,
                           float* output, int hidden, int batch, int steps, bool reverse, int begin, int end,
                           bool use_blas, LstmScratch& scratch, LstmTiming* timing = nullptr) {
    if (timing)
        lstm_direction_impl<true>(projected, recurrent_weight, recurrent_bias, output, hidden, batch, steps, reverse,
                                  begin, end, use_blas, scratch, timing);
    else
        lstm_direction_impl<false>(projected, recurrent_weight, recurrent_bias, output, hidden, batch, steps, reverse,
                                   begin, end, use_blas, scratch, nullptr);
}
} // namespace scnet_detail
