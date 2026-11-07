#include "../src/core/reuse_selective_scan.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

static std::vector<float> values(size_t n, int mod, int shift, float divisor, float offset = 0) {
    std::vector<float> result(n);
    for (size_t i = 0; i < n; ++i)
        result[i] = offset + (static_cast<int>(i % mod) - shift) / divisor;
    return result;
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    std::vector<float> output, state;
    const float gate = 1.0f / (1.0f + std::exp(-1.0f));
    if (!reuse_selective_scan::run(1, 1, 2, 1, {1, 1}, {1, 1}, {0}, {1, 1}, {1, 1}, {0}, {1, 1}, output, state) ||
        std::abs(output[0] - gate) > 1e-6f || std::abs(output[1] - 2 * gate) > 1e-6f || state[0] != 2)
        return 2;
    auto u = values(24, 11, 5, 8);
    auto delta = values(24, 7, 0, 16, 0.125f);
    auto a = values(6, 5, 0, 8, 0.125f);
    for (auto& value : a)
        value = -value;
    auto b = values(16, 9, 4, 8), c = values(16, 7, 3, 8), d = values(3, 5, 2, 8);
    auto z = values(24, 13, 6, 8);
    if (!reuse_selective_scan::run(2, 3, 4, 2, u, delta, a, b, c, d, z, output, state))
        return 3;
    FILE* file = std::fopen(argv[1], "rb");
    if (!file)
        return 4;
    double max_error = 0;
    for (const auto* array : {&output, &state})
        for (float value : *array) {
            unsigned char bytes[4];
            if (std::fread(bytes, 1, 4, file) != 4) {
                std::fclose(file);
                return 5;
            }
            const uint32_t bits =
                uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
            float expected;
            std::memcpy(&expected, &bits, sizeof(expected));
            max_error = std::max(max_error, std::abs(double(value) - expected));
        }
    const bool extra = std::fgetc(file) != EOF;
    std::fclose(file);
    std::printf("RE-USE selective scan: PyTorch max error %.9g\n", max_error);
    if (extra || max_error > 0.00001)
        return 6;
    output = {42};
    state = {43};
    delta[0] = -1;
    if (reuse_selective_scan::run(2, 3, 4, 2, u, delta, a, b, c, d, z, output, state) ||
        output != std::vector<float>{42} || state != std::vector<float>{43})
        return 7;
    delta[0] = 1;
    u[0] = std::numeric_limits<float>::quiet_NaN();
    if (reuse_selective_scan::run(2, 3, 4, 2, u, delta, a, b, c, d, z, output, state))
        return 8;
    u[0] = 0;
    if (reuse_selective_scan::run(2, 3, 4, 2, u, delta, a, b, c, d, z, output, output))
        return 8;
    return 0;
}
