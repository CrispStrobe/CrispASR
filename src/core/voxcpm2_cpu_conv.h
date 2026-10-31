#pragma once

#include "parallel_for.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace core_voxcpm2 {

// A final audio convolution has one output channel and thousands of samples.
// Partition time as well as channels so that it can use the caller's budget.
// Each tile owns disjoint outputs; every individual dot product stays serial.
template <typename F>
inline void for_each_output_tile(int out_ch, int T_out, int n_threads, uint64_t work_per_output, F&& fn) {
    if (out_ch <= 0 || T_out <= 0)
        return;
    constexpr int tile_size = 256;
    const int tiles_per_channel = 1 + (T_out - 1) / tile_size;
    const int64_t tile_count = (int64_t)out_ch * tiles_per_channel;
    if (tile_count > std::numeric_limits<int>::max()) {
        // The chunk helper has an int item count. Extremely large shapes can
        // still partition by channel without overflowing the time-tile count.
        const uint64_t row_work = work_per_output > std::numeric_limits<uint64_t>::max() / (uint64_t)T_out
                                      ? std::numeric_limits<uint64_t>::max()
                                      : (uint64_t)T_out * work_per_output;
        core_parallel::for_each_chunk_work(out_ch, n_threads, row_work, 65536, [&](int begin, int end) {
            for (int oc = begin; oc < end; ++oc)
                fn(oc, 0, T_out);
        });
        return;
    }
    const int n_tiles = (int)tile_count;
    const uint64_t tile_work =
        work_per_output > std::numeric_limits<uint64_t>::max() / (uint64_t)std::min(tile_size, T_out)
            ? std::numeric_limits<uint64_t>::max()
            : (uint64_t)std::min(tile_size, T_out) * work_per_output;
    core_parallel::for_each_chunk_work(n_tiles, n_threads, tile_work, 65536, [&](int begin, int end) {
        for (int tile = begin; tile < end; ++tile) {
            const int oc = tile / tiles_per_channel;
            const int first = (tile % tiles_per_channel) * tile_size;
            fn(oc, first, first + std::min(tile_size, T_out - first));
        }
    });
}

// VoxCPM2's scalar causal Conv1d, shared with the model-free regression tests.
// Shapes are the validated internal model shapes. Weight [out, in/groups, k],
// input [in, time], output [out, time/stride]. n_threads==1 is truly serial,
// including the layout conversions. Reduction order matches the release path.
inline void causal_conv1d(const float* weight, const float* bias, const float* x_in, float* x_out, int out_ch,
                          int in_ch, int ksize, int T_in, int stride, int dilation, int groups, int n_threads) {
    const int T_out = T_in / stride;
    if (T_out <= 0 || out_ch <= 0)
        return;
    const int pad = (ksize - 1) * dilation;
    const int in_per_grp = in_ch / groups;
    const int out_per_grp = out_ch / groups;
    const uint64_t work_per_output = (uint64_t)in_per_grp * ksize;

    if (in_per_grp <= 1 || ksize <= 1) {
        for_each_output_tile(out_ch, T_out, n_threads, work_per_output, [&](int oc, int begin, int end) {
            const int g = oc / out_per_grp;
            for (int ot = begin; ot < end; ++ot) {
                float acc = bias ? bias[oc] : 0.0f;
                const int it_center = ot * stride;
                for (int k = 0; k < ksize; ++k) {
                    const int it = it_center - pad + k * dilation;
                    if (it < 0 || it >= T_in)
                        continue;
                    for (int ic = 0; ic < in_per_grp; ++ic) {
                        const float x = x_in[(size_t)(g * in_per_grp + ic) * T_in + it];
                        const float w = weight[(size_t)oc * in_per_grp * ksize + (size_t)ic * ksize + k];
                        acc += x * w;
                    }
                }
                x_out[(size_t)oc * T_out + ot] = acc;
            }
        });
        return;
    }

    // Keep the release's contiguous inner input-channel dot products.
    std::vector<float> w_kio((size_t)ksize * out_ch * in_per_grp);
    core_parallel::for_each_chunk_work(ksize * out_ch, n_threads, in_per_grp, 65536, [&](int begin, int end) {
        for (int row = begin; row < end; ++row) {
            const int k = row / out_ch;
            const int oc = row % out_ch;
            float* dst = w_kio.data() + (size_t)row * in_per_grp;
            for (int ic = 0; ic < in_per_grp; ++ic)
                dst[ic] = weight[(size_t)oc * in_per_grp * ksize + (size_t)ic * ksize + k];
        }
    });

    std::vector<float> x_tic((size_t)T_in * in_ch);
    core_parallel::for_each_chunk_work(T_in, n_threads, in_ch, 65536, [&](int begin, int end) {
        for (int t = begin; t < end; ++t) {
            float* dst = x_tic.data() + (size_t)t * in_ch;
            for (int ic = 0; ic < in_ch; ++ic)
                dst[ic] = x_in[(size_t)ic * T_in + t];
        }
    });

    for_each_output_tile(out_ch, T_out, n_threads, work_per_output, [&](int oc, int begin, int end) {
        const int g = oc / out_per_grp;
        for (int ot = begin; ot < end; ++ot) {
            float acc = bias ? bias[oc] : 0.0f;
            const int it_center = ot * stride;
            for (int k = 0; k < ksize; ++k) {
                const int it = it_center - pad + k * dilation;
                if (it < 0 || it >= T_in)
                    continue;
                const float* x_row = x_tic.data() + (size_t)it * in_ch + (size_t)g * in_per_grp;
                const float* w_row = w_kio.data() + ((size_t)k * out_ch + oc) * in_per_grp;
                float dot = 0.0f;
                for (int ic = 0; ic < in_per_grp; ++ic)
                    dot += x_row[ic] * w_row[ic];
                acc += dot;
            }
            x_out[(size_t)oc * T_out + ot] = acc;
        }
    });
}

} // namespace core_voxcpm2
