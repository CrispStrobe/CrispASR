#include "../src/core/reuse_tf_mamba.h"
#include <cmath>
#include <cstdio>
#include <fstream>

static bool read(std::ifstream& file, std::vector<float>& out, size_t count) {
    out.resize(count);
    return bool(file.read(reinterpret_cast<char*>(out.data()), count * sizeof(float)));
}
static bool read_mamba(std::ifstream& file, reuse_mamba::Weights& w) {
    w.model = 64;
    w.inner = 256;
    w.rank = 4;
    w.states = 16;
    w.kernel = 4;
    return read(file, w.in_proj, 2 * w.inner * w.model) && read(file, w.conv, w.inner * w.kernel) &&
           read(file, w.conv_bias, w.inner) && read(file, w.x_proj, (w.rank + 2 * w.states) * w.inner) &&
           read(file, w.dt_proj, w.inner * w.rank) && read(file, w.dt_bias, w.inner) &&
           read(file, w.a_log, w.inner * w.states) && read(file, w.d, w.inner) &&
           read(file, w.out_proj, w.model * w.inner);
}
static bool read_bimamba(std::ifstream& file, reuse_bimamba::Weights& w) {
    return read_mamba(file, w.forward) && read_mamba(file, w.backward) && read(file, w.projection, 64 * 2 * 64) &&
           read(file, w.projection_bias, 64) && read(file, w.norm_weight, 64) && read(file, w.norm_bias, 64);
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    std::ifstream file(argv[1], std::ios::binary);
    reuse_tf_mamba::Weights w;
    std::vector<float> expected;
    if (!read_bimamba(file, w.time) || !read_bimamba(file, w.frequency) || !read(file, expected, 64 * 3 * 4) ||
        file.peek() != std::char_traits<char>::eof())
        return 2;
    reuse_dense_encoder::Tensor input{64, 3, 4, std::vector<float>(64 * 3 * 4)}, output;
    for (size_t i = 0; i < input.values.size(); ++i)
        input.values[i] = (static_cast<int>(i % 19) - 9) / 16.0f;
    if (!reuse_tf_mamba::run(w, input, output) || output.values.size() != expected.size())
        return 3;
    double max_error = 0, square = 0, reference_square = 0;
    for (size_t i = 0; i < output.values.size(); ++i) {
        const double diff = double(output.values[i]) - expected[i];
        max_error = std::max(max_error, std::abs(diff));
        square += diff * diff;
        reference_square += double(expected[i]) * expected[i];
    }
    if (reference_square == 0)
        return 4;
    const double relative = std::sqrt(square / reference_square);
    std::printf("RE-USE original-weight time/frequency Mamba: max error %.9g, relative RMSE %.9g\n", max_error,
                relative);
    return max_error <= 0.0003 && relative <= 0.00005 ? 0 : 5;
}
