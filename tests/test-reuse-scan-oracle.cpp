#include "../src/core/reuse_selective_scan.h"
#include <cmath>
#include <cstdio>
#include <fstream>

static bool read(std::ifstream& file, std::vector<float>& out, size_t count) {
    out.resize(count);
    return bool(file.read(reinterpret_cast<char*>(out.data()), count * sizeof(float)));
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    std::ifstream file(argv[1], std::ios::binary);
    std::vector<float> u, delta, a, b, c, d, z, expected, output, state;
    if (!read(file, u, 2 * 256 * 3) || !read(file, delta, 2 * 256 * 3) || !read(file, a, 256 * 16) ||
        !read(file, b, 2 * 3 * 16) || !read(file, c, 2 * 3 * 16) || !read(file, d, 256) ||
        !read(file, z, 2 * 256 * 3) || !read(file, expected, 2 * 256 * 3) ||
        file.peek() != std::char_traits<char>::eof())
        return 2;
    if (!reuse_selective_scan::run(2, 256, 3, 16, u, delta, a, b, c, d, z, output, state))
        return 3;
    double max_error = 0, square = 0, reference_square = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        const double diff = double(output[i]) - expected[i];
        max_error = std::max(max_error, std::abs(diff));
        square += diff * diff;
        reference_square += double(expected[i]) * expected[i];
    }
    if (reference_square == 0)
        return 4;
    const double relative = std::sqrt(square / reference_square);
    std::printf("RE-USE original-weight scan: max error %.9g, relative RMSE %.9g\n", max_error, relative);
    return max_error <= 0.0001 && relative <= 0.00002 ? 0 : 5;
}
