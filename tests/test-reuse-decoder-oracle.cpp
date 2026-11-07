#include "../src/core/reuse_decoder.h"
#include <cmath>
#include <cstdio>
#include <fstream>

static bool read(std::ifstream& file, std::vector<float>& out, size_t count) {
    out.resize(count);
    return bool(file.read(reinterpret_cast<char*>(out.data()), count * sizeof(float)));
}
static bool read_dense(std::ifstream& file, reuse_decoder::Layer& layer, size_t inputs) {
    layer.in_channels = inputs;
    layer.out_channels = 64;
    layer.kernel_time = layer.kernel_freq = 3;
    return read(file, layer.weights, 64 * inputs * 9) && read(file, layer.bias, 64) &&
           read(file, layer.norm_weight, 64) && read(file, layer.norm_bias, 64) && read(file, layer.slope, 64);
}
static bool read_up(std::ifstream& file, reuse_decoder::UpLayer& layer, size_t scale) {
    layer.channels = 64;
    layer.scale = scale;
    return read(file, layer.weights, scale * 64 * 64 * 3) && read(file, layer.bias, scale * 64) &&
           read(file, layer.norm_weight, 64) && read(file, layer.norm_bias, 64) && read(file, layer.slope, 64);
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    std::ifstream file(argv[1], std::ios::binary);
    reuse_decoder::Tensor input{64, 2, 3, std::vector<float>(64 * 2 * 3)};
    for (size_t i = 0; i < input.values.size(); ++i)
        input.values[i] = (static_cast<int>(i % 19) - 9) / 16.0f;
    double max_error = 0, square = 0, reference_square = 0;
    for (int decoder = 0; decoder < 2; ++decoder) {
        std::vector<reuse_decoder::Layer> layers(4);
        for (size_t i = 0; i < layers.size(); ++i)
            if (!read_dense(file, layers[i], 64 * (i + 1)))
                return 2;
        reuse_decoder::UpLayer up_freq, up_time;
        if (!read_up(file, up_freq, 2) || !read_up(file, up_time, 4))
            return 3;
        const size_t heads = decoder == 0 ? 1 : 2;
        std::vector<std::vector<float>> head_weights(heads), head_bias(heads);
        for (size_t i = 0; i < heads; ++i)
            if (!read(file, head_weights[i], 64) || !read(file, head_bias[i], 1))
                return 4;
        reuse_decoder::Tensor dense, first, swapped, expanded, final;
        if (!reuse_decoder::dense(input, layers, dense) || !reuse_decoder::upsample(dense, up_freq, first) ||
            !reuse_decoder::transpose(first, swapped) || !reuse_decoder::upsample(swapped, up_time, expanded) ||
            !reuse_decoder::transpose(expanded, final))
            return 5;
        std::vector<reuse_decoder::Tensor> actual{dense, first, final};
        for (size_t i = 0; i < heads; ++i) {
            reuse_decoder::Tensor head;
            if (!reuse_decoder::pointwise(final, head_weights[i], head_bias[i][0], head))
                return 6;
            actual.push_back(std::move(head));
        }
        for (const auto& stage : actual) {
            std::vector<float> expected;
            if (!read(file, expected, stage.values.size()))
                return 7;
            for (size_t i = 0; i < expected.size(); ++i) {
                const double difference = double(stage.values[i]) - expected[i];
                max_error = std::max(max_error, std::abs(difference));
                square += difference * difference;
                reference_square += double(expected[i]) * expected[i];
            }
        }
    }
    if (file.peek() != std::char_traits<char>::eof() || reference_square == 0)
        return 8;
    const double relative = std::sqrt(square / reference_square);
    std::printf("RE-USE decoder oracle: max error %.9g, relative RMSE %.9g\n", max_error, relative);
    return max_error <= 0.0005 && relative <= 0.00005 ? 0 : 9;
}
