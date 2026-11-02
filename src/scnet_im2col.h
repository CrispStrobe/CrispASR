// SCNet's F32 convolution input preparation. This is a data-layout operation:
// all padding is already explicit in the source graph, and no values are
// rounded, multiplied, or accumulated here.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace scnet_detail {

struct Im2ColShape {
    int times = 0, frequencies = 0, channels = 0;
    int time_kernel = 0, frequency_kernel = 0, frequency_stride = 0;
};

// Source layout is [T,F,C], possibly a frequency view into a larger tensor.
// Destination is contiguous [Kt*Kf*C,Tout,Fout], with the original kernel
// order (channel, frequency, time). Workers own complete output-row blocks;
// they never interleave channel writes within the same output row.
inline void im2col_rows(const float* source, float* destination, const Im2ColShape& shape, size_t frequency_stride,
                        size_t channel_stride, int worker, int workers) {
    const int kernel = shape.time_kernel * shape.frequency_kernel;
    const int columns = shape.channels * kernel;
    // Keep each output block within roughly 12 KiB so the channel-major input
    // traversal reuses destination cache lines before moving to the next block.
    const int block = std::max(1, std::min(32, 3072 / columns));
    const int time_blocks = (shape.times + block - 1) / block;
    const int64_t blocks = int64_t(time_blocks) * shape.frequencies;
    for (int64_t index = worker; index < blocks; index += workers) {
        const int frequency = static_cast<int>(index / time_blocks);
        const int begin = static_cast<int>(index % time_blocks) * block;
        const int count = std::min(block, shape.times - begin);
        for (int channel = 0; channel < shape.channels; ++channel)
            for (int kf = 0; kf < shape.frequency_kernel; ++kf) {
                const float* input = source + size_t(channel) * channel_stride +
                                     size_t(frequency * shape.frequency_stride + kf) * frequency_stride + begin;
                float* output = destination + (size_t(frequency) * shape.times + begin) * columns + channel * kernel +
                                kf * shape.time_kernel;
                if (shape.time_kernel == 1) {
                    for (int time = 0; time < count; ++time)
                        output[size_t(time) * columns] = input[time];
                } else if (shape.time_kernel == 3) {
                    for (int time = 0; time < count; ++time) {
                        float* row = output + size_t(time) * columns;
                        row[0] = input[time];
                        row[1] = input[time + 1];
                        row[2] = input[time + 2];
                    }
                } else {
                    for (int time = 0; time < count; ++time)
                        std::memcpy(output + size_t(time) * columns, input + time,
                                    size_t(shape.time_kernel) * sizeof(float));
                }
            }
    }
}

} // namespace scnet_detail
