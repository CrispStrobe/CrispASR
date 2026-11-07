#include "../src/core/qwen3_ultra_features.h"
#include "../src/core/wav_reader.h"
#include "qwen3_ultra_boundary_fixture.h"
#include "qwen3_ultra_normalizer_fixture.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

int main(int argc, char** argv) {
    if (argc != 3)
        return 1;
    std::vector<float> audio;
    int rate = 0;
    if (!crispasr::core::read_wav_mono_pcm16(argv[1], audio, rate) || rate != 16000)
        return 2;
    FILE* fixture = std::fopen(argv[2], "rb");
    if (!fixture)
        return 3;
    double max_error = 0.0, sum_square = 0.0;
    size_t count = 0, max_index = 0;
    std::array<double, 91> channel_max{};
    for (size_t endpoint = 0; endpoint < 4; ++endpoint) {
        const auto& row = kIntervals[endpoint / 2];
        const double center_ms = row[endpoint % 2];
        qwen3_ultra_features::Features output{};
        if (!qwen3_ultra_features::extract(audio, center_ms, k_mean, k_std, output))
            return 4;
        for (const auto& frame : output) {
            for (float actual : frame) {
                unsigned char bytes[4];
                if (std::fread(bytes, 1, 4, fixture) != 4)
                    return 5;
                uint32_t bits = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) | (uint32_t(bytes[2]) << 16) |
                                (uint32_t(bytes[3]) << 24);
                float expected;
                std::memcpy(&expected, &bits, sizeof(expected));
                const double error = std::abs(static_cast<double>(actual) - expected);
                if (error > max_error) {
                    max_error = error;
                    max_index = count;
                }
                channel_max[count % 91] = std::max(channel_max[count % 91], error);
                sum_square += error * error;
                ++count;
            }
        }
    }
    if (std::fgetc(fixture) != EOF)
        return 6;
    std::fclose(fixture);
    const double rms = std::sqrt(sum_square / count);
    std::printf("Ultra feature reference: %zu values, max error %.9g, RMS error %.9g\n", count, max_error, rms);
    std::printf("max at endpoint %zu frame %zu channel %zu\n", max_index / (41 * 91), (max_index / 91) % 41,
                max_index % 91);
    for (size_t i = 0; i < 91; ++i)
        if (channel_max[i] > 0.003)
            std::printf("channel %zu %.9g\n", i, channel_max[i]);
    if (max_error > 0.0002 || rms > 0.00002)
        return 7;
    qwen3_ultra_features::Features output{};
    auto invalid = k_std;
    invalid[1] = 0;
    if (qwen3_ultra_features::extract(audio, 0, k_mean, invalid, output) ||
        qwen3_ultra_features::extract(audio, std::numeric_limits<double>::infinity(), k_mean, k_std, output))
        return 8;
    std::vector<float> silence(16000, 0.0f);
    if (!qwen3_ultra_features::extract(silence, 0, k_mean, k_std, output))
        return 9;
    silence[0] = std::numeric_limits<float>::quiet_NaN();
    if (qwen3_ultra_features::extract(silence, 0, k_mean, k_std, output))
        return 10;
    std::array<double, 512> real{}, imag{};
    real[0] = 1.0;
    qwen3_ultra_features::fft512(real, imag);
    for (size_t i = 0; i < real.size(); ++i)
        if (std::abs(real[i] - 1.0) > 1e-12 || std::abs(imag[i]) > 1e-12)
            return 11;
    return 0;
}
