#include "../src/core/reuse_dense_encoder.h"
#include <cmath>
#include <cstdio>
#include <fstream>

using namespace reuse_dense_encoder;

static bool read(std::ifstream& f, std::vector<float>& out, size_t count) {
    out.resize(count);
    return bool(f.read(reinterpret_cast<char*>(out.data()), count * sizeof(float)));
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    std::ifstream file(argv[1], std::ios::binary);
    std::vector<Layer> weights(6);
    for (size_t i = 0; i < weights.size(); ++i) {
        auto& w = weights[i];
        w.in_channels = i == 0 ? 2 : i == 5 ? 64 : i * 64;
        w.out_channels = 64;
        w.kernel_time = i == 0 || i == 5 ? 1 : 3;
        w.kernel_freq = i == 0 ? 1 : 3;
        if (!read(file, w.weights, w.out_channels * w.in_channels * w.kernel_time * w.kernel_freq) ||
            !read(file, w.bias, 64) || !read(file, w.norm_weight, 64) || !read(file, w.norm_bias, 64) ||
            !read(file, w.slope, 64))
            return 2;
    }
    Tensor input{2, 5, 7, std::vector<float>(2 * 5 * 7)};
    for (size_t i = 0; i < input.values.size(); ++i)
        input.values[i] = (static_cast<int>(i % 31) - 15) / 32.0f;
    Tensor output;
    std::vector<Tensor> stages;
    if (!encode(input, weights, output, &stages) || stages.size() != 6)
        return 3;
    double max_error = 0, square = 0, reference_square = 0;
    for (const auto& stage : stages) {
        std::vector<float> expected;
        if (!read(file, expected, stage.values.size()))
            return 4;
        for (size_t i = 0; i < expected.size(); ++i) {
            const double diff = double(stage.values[i]) - expected[i];
            max_error = std::max(max_error, std::abs(diff));
            square += diff * diff;
            reference_square += double(expected[i]) * expected[i];
        }
    }
    if (file.peek() != std::char_traits<char>::eof() || reference_square == 0)
        return 5;
    const double relative = std::sqrt(square / reference_square);
    std::printf("RE-USE original dense encoder: max error %.9g, relative RMSE %.9g\n", max_error, relative);
    return max_error <= 0.0002 && relative <= 0.00002 ? 0 : 6;
}
