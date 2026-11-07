// Experimental RE-USE CPU frontend; see frontend_mkl_probe.md.
// Independently written against Intel's public oneMKL VML/DFTI interfaces.
// No PyTorch implementation or private PyTorch symbols are used.
#include "../../../src/core/wav_reader.h"
#include "../../../src/core/reuse_frontend_mkl.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int fft_size = 640, hop = 80, bins = 321;
using reuse_frontend_mkl::Analysis;
using reuse_frontend_mkl::transform;
using reuse_frontend_mkl::Window;

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
            const auto result = transform(audio, Window::HighAccuracy, true);
            for (size_t frame = 0; frame < 21; ++frame) {
                const size_t i = frame * bins + (alternating ? bins - 1 : 0);
                if (result.phase[i] != (sign < 0 ? pi : 0.0f) || result.imaginary[i] != 0.0f ||
                    std::signbit(result.imaginary[i]) || std::abs(result.real[i] - sign * 80.0f) > 1e-4f)
                    return false;
                ++checks;
            }
        }
    }
    const auto silence = transform(std::vector<float>(1600), Window::HighAccuracy, true);
    for (float value : silence.magnitude)
        if (value != 0.0f)
            return false;
    std::printf("REUSE_MKL_SELF_TEST_PASS exact_endpoint_and_scale_checks=%zu silence_bins=%zu\n", checks,
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
        std::printf("REUSE_MKL_VERSION %s\n", reuse_frontend_mkl::version().c_str());
        if (argc == 2 && std::string(argv[1]) == "--self-test")
            return self_test() ? 0 : 1;
        if (argc < 3 || argc > 5)
            throw std::runtime_error("usage: frontend-mkl-probe input.wav output.f32 [ha|la|ep|libm] [batch|single]");
        const std::string mode = argc > 3 ? argv[3] : "ha", layout = argc > 4 ? argv[4] : "batch";
        if ((mode != "ha" && mode != "la" && mode != "ep" && mode != "libm") ||
            (layout != "batch" && layout != "single"))
            throw std::runtime_error("invalid window mode or transform layout");
        std::vector<float> audio;
        int rate = 0;
        if (!mono_header(argv[1]) || !crispasr::core::read_wav_mono_pcm16(argv[1], audio, rate) || rate != 16000)
            throw std::runtime_error("expected a readable mono PCM16 16 kHz WAV");
        const Window accuracy = mode == "ha"   ? Window::HighAccuracy
                                : mode == "la" ? Window::LowAccuracy
                                : mode == "ep" ? Window::EnhancedPerformance
                                               : Window::Libm;
        const auto result = transform(audio, accuracy, layout == "batch");
        if (!write(argv[2], result))
            throw std::runtime_error("failed to write complete frontend diagnostic");
        std::printf("REUSE_MKL_FRONTEND_WRITTEN samples=%zu frames=%zu window=%s layout=%s\n", audio.size(),
                    1 + audio.size() / hop, mode.c_str(), layout.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "REUSE_MKL_FRONTEND_ERROR %s\n", error.what());
        return 1;
    }
}
