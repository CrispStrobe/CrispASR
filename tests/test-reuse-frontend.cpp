#include "../src/core/reuse_frontend.h"
#include "../src/core/wav_reader.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

static bool compare_reference(const std::vector<float>& audio, int rate, const char* fixture_path, int expected_fft,
                              int expected_hop) {
    reuse_frontend::Spectrum out;
    if (!reuse_frontend::analyze(audio, rate, out) || out.n_fft != expected_fft || out.hop != expected_hop ||
        out.frames != 1 + audio.size() / static_cast<size_t>(expected_hop) ||
        out.bins != static_cast<size_t>(expected_fft / 2 + 1) || out.magnitude.size() != out.frames * out.bins ||
        out.phase.size() != out.frames * out.bins)
        return false;
    FILE* fixture = std::fopen(fixture_path, "rb");
    if (!fixture)
        return false;
    std::vector<float> ref_mag(out.magnitude.size()), ref_phase(out.phase.size());
    for (auto* values : {&ref_mag, &ref_phase}) {
        for (auto& value : *values) {
            unsigned char bytes[4];
            if (std::fread(bytes, 1, 4, fixture) != 4) {
                std::fclose(fixture);
                return false;
            }
            const uint32_t bits = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) | (uint32_t(bytes[2]) << 16) |
                                  (uint32_t(bytes[3]) << 24);
            std::memcpy(&value, &bits, sizeof(value));
        }
    }
    const bool extra = std::fgetc(fixture) != EOF;
    std::fclose(fixture);
    if (extra)
        return false;
    double max_mag = 0.0, max_phase = 0.0, squared = 0.0, baseline = 0.0;
    for (size_t i = 0; i < ref_mag.size(); ++i) {
        max_mag = std::max(max_mag, std::abs(double(out.magnitude[i]) - ref_mag[i]));
        const double diff = double(out.phase[i]) - ref_phase[i];
        if (std::expm1(ref_mag[i]) > 0.01)
            max_phase = std::max(max_phase, std::abs(std::atan2(std::sin(diff), std::cos(diff))));
        const double expected_mag = std::expm1(ref_mag[i]);
        const double actual_mag = std::expm1(out.magnitude[i]);
        const double re = actual_mag * std::cos(out.phase[i]) - expected_mag * std::cos(ref_phase[i]);
        const double im = actual_mag * std::sin(out.phase[i]) - expected_mag * std::sin(ref_phase[i]);
        squared += re * re + im * im;
        baseline += expected_mag * expected_mag;
    }
    const double relative = std::sqrt(squared / baseline);
    std::printf("RE-USE %d Hz frontend: max magnitude error %.9g, max phase error %.9g, relative error %.9g\n", rate,
                max_mag, max_phase, relative);
    return max_mag <= 0.0001 && max_phase <= 0.001 && relative <= 0.00001;
}

int main(int argc, char** argv) {
    if (argc != 4)
        return 1;
    std::vector<float> audio;
    int rate = 0;
    if (!crispasr::core::read_wav_mono_pcm16(argv[1], audio, rate) || rate != 16000 || audio.size() < 1600)
        return 2;
    audio.resize(1600);
    if (!compare_reference(audio, rate, argv[2], 640, 80))
        return 3;
    std::vector<float> synthetic(4400);
    for (int i = 0; i < static_cast<int>(synthetic.size()); ++i)
        synthetic[i] = static_cast<float>((i * 37) % 2001 - 1000) / 32768.0f;
    if (!compare_reference(synthetic, 44100, argv[3], 1764, 220))
        return 4;
    reuse_frontend::Spectrum untouched;
    untouched.sample_rate = 123;
    if (reuse_frontend::analyze(audio, 7999, untouched) || untouched.sample_rate != 123 ||
        reuse_frontend::analyze(std::vector<float>(320), 16000, untouched))
        return 5;
    audio[0] = std::numeric_limits<float>::quiet_NaN();
    if (reuse_frontend::analyze(audio, 16000, untouched))
        return 6;
    std::vector<float> silence(2000, 0.0f);
    reuse_frontend::Spectrum out;
    if (!reuse_frontend::analyze(silence, 44100, out) || out.n_fft != 1764 || out.hop != 220)
        return 7;
    for (float x : out.magnitude)
        if (x != 0.0f)
            return 8;
    return 0;
}
