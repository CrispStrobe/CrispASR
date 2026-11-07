#include "../src/core/reuse_dense_encoder.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace reuse_dense_encoder;

static std::vector<float> values(size_t n, int mod, int shift, float divisor, float offset = 0) {
    std::vector<float> result(n);
    for (size_t i = 0; i < n; ++i)
        result[i] = offset + (static_cast<int>(i % mod) - shift) / divisor;
    return result;
}
static Layer layer(size_t in, size_t kt, size_t kf) {
    Layer w;
    w.in_channels = in;
    w.out_channels = 64;
    w.kernel_time = kt;
    w.kernel_freq = kf;
    w.weights = values(64 * in * kt * kf, 17, 8, 128);
    w.bias = values(64, 7, 3, 64);
    w.norm_weight = values(64, 9, 4, 64, 1);
    w.norm_bias = values(64, 11, 5, 128);
    w.slope = values(64, 5, 2, 64, 0.25f);
    return w;
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    Tensor analytic{1, 1, 2, {-1.0f, 1.0f}}, analytic_out;
    Layer identity{1, 1, 1, 1, {1.0f}, {0.0f}, {1.0f}, {0.0f}, {0.5f}};
    const float expected_scale = 1.0f / std::sqrt(1.0f + 1e-5f);
    if (!conv_norm(analytic, identity, 1, 1, 1, 0, 0, analytic_out) ||
        std::abs(analytic_out.values[0] + 0.5f * expected_scale) > 1e-6f ||
        std::abs(analytic_out.values[1] - expected_scale) > 1e-6f ||
        conv_norm(analytic, identity, 0, 1, 1, 0, 0, analytic_out))
        return 8;
    Tensor input{2, 5, 7, values(2 * 5 * 7, 31, 15, 32)};
    std::vector<Layer> weights{layer(2, 1, 1),   layer(64, 3, 3),  layer(128, 3, 3),
                               layer(192, 3, 3), layer(256, 3, 3), layer(64, 1, 3)};
    Tensor output;
    std::vector<Tensor> stages;
    if (!encode(input, weights, output, &stages) || stages.size() != 6 || output.channels != 64 || output.time != 2 ||
        output.freq != 4)
        return 2;
    FILE* ref = std::fopen(argv[1], "rb");
    if (!ref)
        return 3;
    double max_error = 0;
    for (const auto& stage : stages)
        for (float value : stage.values) {
            unsigned char bytes[4];
            if (std::fread(bytes, 1, 4, ref) != 4) {
                std::fclose(ref);
                return 4;
            }
            const uint32_t bits =
                uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
            float expected;
            std::memcpy(&expected, &bits, sizeof(expected));
            max_error = std::max(max_error, std::abs(double(value) - expected));
        }
    const bool extra = std::fgetc(ref) != EOF;
    std::fclose(ref);
    std::printf("RE-USE dense encoder: 6 PyTorch stages, max error %.9g\n", max_error);
    if (extra || max_error > 0.0002)
        return 5;
    output.values.assign(1, 42.0f);
    weights[2].weights.pop_back();
    if (encode(input, weights, output) || output.values != std::vector<float>{42.0f})
        return 6;
    weights[2] = layer(128, 3, 3);
    input.values[0] = std::numeric_limits<float>::quiet_NaN();
    if (encode(input, weights, output) || output.values != std::vector<float>{42.0f})
        return 7;
    return 0;
}
