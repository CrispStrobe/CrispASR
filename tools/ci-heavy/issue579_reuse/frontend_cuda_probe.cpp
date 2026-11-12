// Optional cuFFT research frontend; see frontend_cuda_probe.md.
// Public CUDA host APIs and independent CPU Hann; no PyTorch symbols or reference arrays.
#include "../../../src/core/wav_reader.h"
#include "../../../src/core/reuse_frontend_cuda.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int fft_size = 640, hop = 80, bins = 321;
using reuse_frontend_cuda::Analysis;
using reuse_frontend_cuda::transform;

// The shared PCM reader downmixes channels. Reject that transformation here so
// the diagnostic and Python reference consume identical input sample bytes.
bool mono_header(const char* path) {
    std::ifstream file(path, std::ios::binary);
    char header[12];
    if (!file.read(header, sizeof(header)) || std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4))
        return false;
    unsigned char chunk[8];
    while (file.read(reinterpret_cast<char*>(chunk), sizeof(chunk))) {
        const uint32_t size =
            uint32_t(chunk[4]) | (uint32_t(chunk[5]) << 8) | (uint32_t(chunk[6]) << 16) | (uint32_t(chunk[7]) << 24);
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            unsigned char format[16];
            return size >= sizeof(format) && file.read(reinterpret_cast<char*>(format), sizeof(format)) &&
                   format[0] == 1 && format[1] == 0 && format[2] == 1 && format[3] == 0 && format[14] == 16 &&
                   format[15] == 0;
        }
        file.seekg(static_cast<std::streamoff>(size) + size % 2, std::ios::cur);
    }
    return false;
}

bool self_test() {
    const float pi = static_cast<float>(std::acos(-1.0));
    size_t checks = 0;
    for (bool alternating : {false, true}) {
        for (float sign : {-1.0f, 1.0f}) {
            std::vector<float> audio(1600);
            for (size_t i = 0; i < audio.size(); ++i)
                audio[i] = sign * 0.25f * (alternating && i % 2 ? -1.0f : 1.0f);
            const auto result = transform(audio);
            for (size_t frame = 0; frame < 21; ++frame) {
                const size_t i = frame * bins + (alternating ? bins - 1 : 0);
                if (result.phase[i] != (sign < 0 ? pi : 0.0f) || result.imaginary[i] != 0.0f ||
                    std::signbit(result.imaginary[i]) || std::abs(result.real[i] - sign * 80.0f) > 1e-4f)
                    return false;
                ++checks;
            }
        }
    }
    const auto silence = transform(std::vector<float>(1600));
    for (float value : silence.magnitude)
        if (value != 0.0f)
            return false;
    std::printf("REUSE_CUDA_SELF_TEST_PASS exact_endpoint_and_scale_checks=%zu silence_bins=%zu\n", checks,
                silence.magnitude.size());
    return true;
}

bool write(const char* path, const Analysis& result) {
    FILE* file = std::fopen(path, "wb");
    if (!file)
        return false;
    bool ok = true;
    for (const auto* values : {&result.window, &result.real, &result.imaginary, &result.magnitude, &result.phase}) {
        for (float value : *values) {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            const unsigned char bytes[4] = {static_cast<unsigned char>(bits), static_cast<unsigned char>(bits >> 8),
                                            static_cast<unsigned char>(bits >> 16),
                                            static_cast<unsigned char>(bits >> 24)};
            ok = std::fwrite(bytes, 1, 4, file) == 4 && ok;
        }
    }
    return std::fclose(file) == 0 && ok;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--invalid-self-test") {
            reuse_frontend::Spectrum output;
            output.sample_rate = 123;
            reuse_frontend::Diagnostics observations;
            observations.window = {42};
            for (float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
                if (reuse_frontend_cuda::analyze(std::vector<float>(400, invalid), 16000, output, &observations) ||
                    output.sample_rate != 123 || observations.window != std::vector<float>{42})
                    return 1;
            }
            if (reuse_frontend_cuda::analyze(std::vector<float>(400), 8000, output, &observations) ||
                reuse_frontend_cuda::analyze(std::vector<float>(320), 16000, output, &observations) ||
                output.sample_rate != 123 || observations.window != std::vector<float>{42})
                return 1;
            std::puts("REUSE_CUDA_INVALID_INPUT_OWNERSHIP_PASS");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            const bool passed = self_test();
            std::printf("REUSE_CUDA_VERSION %s\n", reuse_frontend_cuda::version().c_str());
            return passed ? 0 : 1;
        }
        if (argc != 3)
            throw std::runtime_error("usage: frontend-cuda-probe input.wav output.f32");
        std::vector<float> audio;
        int rate = 0;
        if (!mono_header(argv[1]) || !crispasr::core::read_wav_mono_pcm16(argv[1], audio, rate) || rate != 16000)
            throw std::runtime_error("input requires a readable mono PCM16 16 kHz WAV");
        const auto result = transform(audio);
        const auto identity = reuse_frontend_cuda::version();
        if (!write(argv[2], result))
            throw std::runtime_error("failed to write complete frontend diagnostic");
        std::printf("REUSE_CUDA_VERSION %s\n", identity.c_str());
        std::printf("REUSE_CUDA_FRONTEND_WRITTEN samples=%zu frames=%zu\n", audio.size(), 1 + audio.size() / hop);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "REUSE_CUDA_FRONTEND_ERROR %s\n", error.what());
        return 1;
    }
}
