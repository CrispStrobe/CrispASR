#include "../src/core/reuse_decoder.h"
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
static reuse_dense_encoder::Layer dense_layer(size_t inputs) {
    reuse_dense_encoder::Layer layer;
    layer.in_channels = inputs;
    layer.out_channels = 4;
    layer.kernel_time = layer.kernel_freq = 3;
    layer.weights = values(4 * inputs * 3 * 3, 17, 8, 128);
    layer.bias = values(4, 7, 3, 64);
    layer.norm_weight = values(4, 9, 4, 64, 1);
    layer.norm_bias = values(4, 11, 5, 128);
    layer.slope = values(4, 5, 2, 64, 0.25f);
    return layer;
}
static reuse_decoder::UpLayer up_layer(size_t scale) {
    reuse_decoder::UpLayer layer;
    layer.channels = 4;
    layer.scale = scale;
    layer.weights = values(scale * 4 * 4 * 3, 13, 6, 64);
    layer.bias = values(scale * 4, 7, 3, 64);
    layer.norm_weight = values(4, 9, 4, 64, 1);
    layer.norm_bias = values(4, 11, 5, 128);
    layer.slope = values(4, 5, 2, 64, 0.25f);
    return layer;
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    reuse_decoder::Tensor input{4, 2, 3, values(4 * 2 * 3, 19, 9, 16)};
    const std::vector<reuse_decoder::Layer> layers{dense_layer(4), dense_layer(8), dense_layer(12), dense_layer(16)};
    auto up_freq = up_layer(2), up_time = up_layer(4);
    reuse_decoder::Tensor dense, first, transposed, second, complete;
    if (!reuse_decoder::dense(input, layers, dense) || !reuse_decoder::upsample(dense, up_freq, first) ||
        !reuse_decoder::transpose(first, transposed) || !reuse_decoder::upsample(transposed, up_time, second) ||
        !reuse_decoder::transpose(second, complete))
        return 2;
    reuse_decoder::Tensor pipeline;
    if (!reuse_decoder::decode_features(input, layers, up_freq, up_time, pipeline) ||
        pipeline.values != complete.values)
        return 3;
    std::vector<reuse_decoder::Tensor> stages{dense, first, complete};
    for (int shift : {2, 3, 4}) {
        reuse_decoder::Tensor result;
        if (!reuse_decoder::pointwise(complete, values(4, 9, shift, 32), 0.125f, result))
            return 4;
        stages.push_back(std::move(result));
    }
    FILE* fixture = std::fopen(argv[1], "rb");
    if (!fixture)
        return 5;
    double max_error = 0, square = 0, reference_square = 0;
    for (const auto& stage : stages)
        for (float actual : stage.values) {
            unsigned char bytes[4];
            if (std::fread(bytes, 1, 4, fixture) != 4) {
                std::fclose(fixture);
                return 6;
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
        return 7;
    const double relative = std::sqrt(square / reference_square);
    std::printf("RE-USE decoder: max error %.9g, relative RMSE %.9g\n", max_error, relative);
    if (max_error > 0.0001 || relative > 0.00002)
        return 8;
    pipeline.values = {42};
    up_freq.weights.pop_back();
    if (reuse_decoder::decode_features(input, layers, up_freq, up_time, pipeline) ||
        pipeline.values != std::vector<float>{42})
        return 9;
    if (reuse_decoder::pointwise(input, values(4, 9, 2, 32), std::numeric_limits<float>::quiet_NaN(), pipeline))
        return 10;
    return 0;
}
