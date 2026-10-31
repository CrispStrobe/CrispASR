#include <catch2/catch_test_macros.hpp>

#include "core/voxcpm2_cpu_conv.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

TEST_CASE("VoxCPM2 causal convolution preserves grouped and final-audio outputs", "[unit][voxcpm2][cpu-conv]") {
    struct shape {
        int in, out, k, time, stride, dilation, groups;
    };
    const shape shapes[] = {
        {1, 1, 1, 1, 1, 1, 1},    {4, 4, 7, 5, 1, 3, 4},       {4, 6, 3, 17, 2, 2, 2},     {8, 4, 1, 31, 3, 1, 1},
        {12, 6, 7, 513, 1, 3, 3}, {16, 16, 7, 1025, 1, 9, 16}, {32, 1, 7, 48000, 1, 1, 1}, {8, 4, 3, 2, 4, 1, 2},
    };
    for (const auto& s : shapes) {
        INFO("in=" << s.in << " out=" << s.out << " k=" << s.k << " time=" << s.time << " stride=" << s.stride
                   << " dilation=" << s.dilation << " groups=" << s.groups);
        const int T_out = s.time / s.stride;
        const int in_group = s.in / s.groups;
        const int out_group = s.out / s.groups;
        std::vector<float> input((size_t)s.in * s.time), weight((size_t)s.out * in_group * s.k), bias(s.out);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = (float)((int)(i % 29) - 14) / 29.0f;
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = (float)((int)(i % 13) - 6) / 37.0f;
        for (int i = 0; i < s.out; ++i)
            bias[i] = (float)(i - 2) / 11.0f;
        for (bool with_bias : {false, true}) {
            const float* b = with_bias ? bias.data() : nullptr;
            std::vector<float> serial((size_t)s.out * T_out, std::numeric_limits<float>::quiet_NaN());
            auto parallel = serial;
            core_voxcpm2::causal_conv1d(weight.data(), b, input.data(), serial.data(), s.out, s.in, s.k, s.time,
                                        s.stride, s.dilation, s.groups, 1);
            core_voxcpm2::causal_conv1d(weight.data(), b, input.data(), parallel.data(), s.out, s.in, s.k, s.time,
                                        s.stride, s.dilation, s.groups, 4);
            REQUIRE(serial == parallel);
            // Independent mathematical oracle, using double accumulation in
            // natural channel-major order rather than the candidate's layout.
            bool finite = true;
            double max_error = 0.0;
            for (int oc = 0; oc < s.out; ++oc) {
                const int group = oc / out_group;
                for (int ot = 0; ot < T_out; ++ot) {
                    double expected = with_bias ? bias[oc] : 0.0;
                    for (int ic = 0; ic < in_group; ++ic) {
                        for (int k = 0; k < s.k; ++k) {
                            const int ti = ot * s.stride - (s.k - 1) * s.dilation + k * s.dilation;
                            if (ti >= 0 && ti < s.time)
                                expected += (double)input[(size_t)(group * in_group + ic) * s.time + ti] *
                                            weight[((size_t)oc * in_group + ic) * s.k + k];
                        }
                    }
                    const float actual = parallel[(size_t)oc * T_out + ot];
                    finite &= std::isfinite(actual);
                    max_error = std::max(max_error, std::abs(expected - actual));
                }
            }
            REQUIRE(finite);
            REQUIRE(max_error < 1e-5);
        }
    }
}

TEST_CASE("VoxCPM2 convolution partitions the time axis for one output channel", "[unit][voxcpm2][cpu-conv]") {
    // The regression scheduled only out_ch tasks: the final mono convolution
    // therefore had one task at every duration. Check ownership independently
    // of machine speed or the number of CPUs available to this test process.
    const int T = 1025;
    std::vector<int> touched(T, 0);
    core_voxcpm2::for_each_output_tile(1, T, 1, 224, [&](int oc, int begin, int end) {
        REQUIRE(oc == 0);
        REQUIRE(end - begin <= 256);
        for (int t = begin; t < end; ++t)
            ++touched[t];
    });
    REQUIRE(std::all_of(touched.begin(), touched.end(), [](int n) { return n == 1; }));
}

TEST_CASE("VoxCPM2 output scheduling bounds very large tile counts", "[unit][voxcpm2][cpu-conv]") {
    // Scheduling only: no large audio allocation. This shape exceeds the
    // chunk helper's int item count and must use the safe channel fallback.
    int channels = 0;
    core_voxcpm2::for_each_output_tile(257, std::numeric_limits<int>::max(), 1, 224, [&](int oc, int begin, int end) {
        REQUIRE(oc == channels++);
        REQUIRE(begin == 0);
        REQUIRE(end == std::numeric_limits<int>::max());
    });
    REQUIRE(channels == 257);
}
