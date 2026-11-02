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
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

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
    const float* input = storage.data() + view_begin * input_time;
    const int columns = channels * kt * kf;
    std::vector<float> expected(size_t(times) * frequencies * columns);
    for (int f = 0; f < frequencies; ++f)
        for (int t = 0; t < times; ++t)
            for (int c = 0; c < channels; ++c)
                for (int y = 0; y < kf; ++y)
                    for (int x = 0; x < kt; ++x)
                        expected[(size_t(f) * times + t) * columns + (c * kf + y) * kt + x] =
                            input[size_t(c) * channel_stride + (f * sf + y) * input_time + t + x];
    const float untouched = 1234567.0f;
    std::vector<float> output(expected.size(), untouched);
    std::vector<unsigned char> owners(expected.size(), 0);
    // Verify exclusive worker ownership as well as the final packed values.
    for (int worker = 0; worker < workers; ++worker) {
        std::fill(output.begin(), output.end(), untouched);
        scnet_detail::im2col_rows(input, output.data(), shape, input_time, channel_stride, worker, workers);
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
            scnet_detail::im2col_rows(input, output.data(), shape, input_time, channel_stride, worker, workers);
        });
    for (auto& thread : pool) thread.join();
    assert(std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0);
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
        check(614, 3, 5, 2, 5, 2, workers);
        check(1, 1, 1, 1, 1, 1, workers);
    }
    std::cout << "36 packing geometries/worker counts: exact values and exclusive ownership\n";
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
