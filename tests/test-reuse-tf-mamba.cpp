#include "../src/core/reuse_tf_mamba.h"
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
static reuse_bimamba::Weights bidirectional() {
    reuse_bimamba::Weights w;
    w.forward = mamba();
    w.backward = mamba();
    w.projection = values(4 * 2 * 4, 13, 6, 32);
    w.projection_bias = values(4, 7, 3, 32);
    w.norm_weight = values(4, 5, 2, 32, 1);
    w.norm_bias = values(4, 9, 4, 64);
    return w;
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    reuse_tf_mamba::Weights w{bidirectional(), bidirectional()};
    reuse_dense_encoder::Tensor input{4, 5, 3, values(4 * 5 * 3, 19, 9, 16)}, output;
    if (!reuse_tf_mamba::run(w, input, output) || output.channels != 4 || output.time != 5 || output.freq != 3)
        return 2;
    FILE* fixture = std::fopen(argv[1], "rb");
    if (!fixture)
        return 3;
    double max_error = 0, square = 0, reference_square = 0;
    for (float actual : output.values) {
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
    std::printf("RE-USE time/frequency Mamba: max error %.9g, relative RMSE %.9g\n", max_error, relative);
    if (max_error > 0.0001 || relative > 0.00002)
        return 6;
    output.values = {42};
    input.values.pop_back();
    if (reuse_tf_mamba::run(w, input, output) || output.values != std::vector<float>{42})
        return 7;
    return 0;
}
