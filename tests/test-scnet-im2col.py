#!/usr/bin/env python3
"""Check CPU convolution packing against explicit original kernel ordering."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "scnet_im2col.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using Packing = void (*)(const float*, float*, const scnet_detail::Im2ColShape&, size_t, size_t, int, int);

static std::vector<float> check_workers(Packing pack, const float* input, const scnet_detail::Im2ColShape& shape,
                                      size_t input_time, size_t channel_stride, int workers,
                                      const std::vector<float>& expected) {
    const float untouched = 1234567.0f;
    std::vector<float> output(expected.size(), untouched);
    std::vector<unsigned char> owners(expected.size(), 0);
    // Verify exclusive worker ownership as well as the final packed values.
    for (int worker = 0; worker < workers; ++worker) {
        std::fill(output.begin(), output.end(), untouched);
        pack(input, output.data(), shape, input_time, channel_stride, worker, workers);
        for (size_t i = 0; i < output.size(); ++i)
            if (output[i] != untouched) {
                assert(owners[i] == 0);
                ++owners[i];
                assert(std::memcmp(&output[i], &expected[i], sizeof(float)) == 0);
            }
    }
    for (auto owner : owners) assert(owner == 1);
    std::fill(output.begin(), output.end(), untouched);
    std::vector<std::thread> pool;
    for (int worker = 0; worker < workers; ++worker)
        pool.emplace_back([&, worker] {
            pack(input, output.data(), shape, input_time, channel_stride, worker, workers);
        });
    for (auto& thread : pool) thread.join();
    assert(std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0);
    return output;
}

static void check(int times, int frequencies, int channels, int kt, int kf, int sf, int workers) {
    scnet_detail::Im2ColShape shape{times, frequencies, channels, kt, kf, sf};
    const int input_time = times + kt - 1;
    const int input_frequency = (frequencies - 1) * sf + kf;
    // An interior frequency view retains the parent channel stride.
    const int view_begin = 3, parent_frequency = input_frequency + 9;
    const size_t channel_stride = size_t(input_time) * parent_frequency;
    std::vector<float> storage(channel_stride * channels);
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = float(int((i * 127 + 11) % 10007) - 5003) * .125f;
    storage[view_begin * input_time] = -0.0f;
    if (input_time > 1) {
        const uint32_t payload = 0x7fc12345U;
        std::memcpy(storage.data() + view_begin * input_time + 1, &payload, sizeof(float));
    }
    const float* input = storage.data() + view_begin * input_time;
    const int columns = channels * kt * kf;
    const size_t positions = size_t(times) * frequencies;
    // Independent index oracle follows the source convolution axes, rather
    // than either production helper's worker decomposition or memcpy loops.
    std::vector<float> expected(positions * columns);
    for (int f = 0; f < frequencies; ++f)
        for (int t = 0; t < times; ++t)
            for (int c = 0; c < channels; ++c)
                for (int y = 0; y < kf; ++y)
                    for (int x = 0; x < kt; ++x)
                        expected[(size_t(f) * times + t) * columns + (c * kf + y) * kt + x] =
                            input[size_t(c) * channel_stride + (f * sf + y) * input_time + t + x];
    std::vector<float> transposed(expected.size());
    for (size_t position = 0; position < positions; ++position)
        for (int column = 0; column < columns; ++column)
            std::memcpy(&transposed[size_t(column) * positions + position],
                        &expected[position * columns + column], sizeof(float));
    const auto rows = check_workers(scnet_detail::im2col_rows, input, shape, input_time,
                                   channel_stride, workers, expected);
    const auto cols = check_workers(scnet_detail::im2col_columns, input, shape, input_time,
                                   channel_stride, workers, transposed);
    // In addition to each independent oracle, require the two production
    // layouts to be byte-exact matrix transposes, including signed zero/NaN.
    for (size_t position = 0; position < positions; ++position)
        for (int column = 0; column < columns; ++column)
            assert(std::memcmp(&cols[size_t(column) * positions + position],
                               &rows[position * columns + column], sizeof(float)) == 0);
}

int main() {
    for (int workers : {1, 2, 3, 8}) {
        check(13, 7, 4, 1, 3, 1, workers);
        check(17, 5, 32, 1, 4, 4, workers);
        check(11, 3, 64, 1, 16, 16, workers);
        check(15, 4, 8, 3, 1, 1, workers);
        check(19, 9, 16, 1, 1, 1, workers);
        check(23, 7, 128, 3, 3, 1, workers);
        check(872, 2, 32, 3, 3, 1, workers);
        check(872, 32, 4, 1, 3, 1, workers);
        check(614, 17, 8, 1, 1, 1, workers);
        check(194, 3, 32, 3, 3, 1, workers);
        check(614, 3, 5, 2, 5, 2, workers);
        check(1, 1, 1, 1, 1, 1, workers);
    }
    std::cout << "48 packing geometries/worker counts, both layouts: exact values, byte transposes and exclusive ownership\n";
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix="scnet-im2col-") as temporary:
        source = Path(temporary) / "check.cpp"
        binary = Path(temporary) / "check"
        source.write_text(SOURCE)
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror",
                        "-pthread", "-I", str(ROOT / "src"), str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
