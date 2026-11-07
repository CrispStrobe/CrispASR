#include "../src/core/reuse_bimamba.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static std::vector<float> values(size_t n, int mod, int shift, float divisor, float offset = 0) {
    std::vector<float> result(n);
    for (size_t i = 0; i < n; ++i)
        result[i] = offset + (static_cast<int>(i % mod) - shift) / divisor;
    return result;
}
static reuse_mamba::Weights mamba() {
    reuse_mamba::Weights w;
    w.model = 4;
    w.inner = 8;
    w.rank = 1;
    w.states = 2;
    w.kernel = 3;
    w.in_proj = values(2 * 8 * 4, 13, 6, 64);
    w.conv = values(8 * 3, 11, 5, 32);
    w.conv_bias = values(8, 7, 3, 64);
    w.x_proj = values((1 + 2 * 2) * 8, 17, 8, 64);
    w.dt_proj = values(8, 7, 3, 32);
    w.dt_bias = values(8, 9, 4, 32);
    w.a_log = values(8 * 2, 5, 2, 16);
    w.d = values(8, 7, 3, 32);
    w.out_proj = values(4 * 8, 11, 5, 64);
    return w;
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    reuse_bimamba::Weights w;
    w.forward = mamba();
    w.backward = mamba();
    w.projection = values(4 * 2 * 4, 13, 6, 32);
    w.projection_bias = values(4, 7, 3, 32);
    w.norm_weight = values(4, 5, 2, 32, 1);
    w.norm_bias = values(4, 9, 4, 64);
    auto input = values(2 * 5 * 4, 17, 8, 16);
    std::vector<float> output;
    if (!reuse_bimamba::run(w, 2, 5, input, output))
        return 2;
    FILE* fixture = std::fopen(argv[1], "rb");
    if (!fixture)
        return 3;
    double max_error = 0, square = 0, reference_square = 0;
    for (float actual : output) {
        unsigned char bytes[4];
        if (std::fread(bytes, 1, 4, fixture) != 4) {
            std::fclose(fixture);
            return 4;
        }
        const uint32_t bits =
            uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
        float expected;
        std::memcpy(&expected, &bits, sizeof(expected));
        const double error = double(actual) - expected;
        max_error = std::max(max_error, std::abs(error));
        square += error * error;
        reference_square += double(expected) * expected;
    }
    const bool extra = std::fgetc(fixture) != EOF;
    std::fclose(fixture);
    if (extra || reference_square == 0)
        return 5;
    const double relative = std::sqrt(square / reference_square);
    std::printf("RE-USE bidirectional Mamba: max error %.9g, relative RMSE %.9g\n", max_error, relative);
    if (max_error > 0.0001 || relative > 0.00002)
        return 6;
    auto changed = input;
    for (size_t i = 16; i < 20; ++i)
        changed[i] += static_cast<float>(i - 15) * 1024.0f;
    std::vector<float> influenced;
    if (!reuse_bimamba::run(w, 2, 5, changed, influenced))
        return 7;
    bool saw_backward = false;
    for (size_t i = 0; i < 4; ++i)
        saw_backward = saw_backward || std::abs(output[i] - influenced[i]) > 1e-8f;
    if (!saw_backward) {
        std::fprintf(stderr, "future input did not influence the first frame\n");
        return 8;
    }
    output = {42};
    w.projection.pop_back();
    if (reuse_bimamba::run(w, 2, 5, input, output) || output != std::vector<float>{42})
        return 9;
    return 0;
}
