// Same-binary serial/parallel projection probe; orchestration retains raw timings.
#include "core/fft.h"
#include "core/mel.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 7)
        return 2;
    const int frames = std::atoi(argv[1]);
    core_mel::Params p;
    p.n_fft = p.win_length = 512;
    p.hop_length = 160;
    p.n_mels = 80;
    p.center_pad = false;
    p.log_base = core_mel::LogBase::None;
    p.norm = core_mel::Normalization::None;
    p.layout = core_mel::Layout::TimeMels;
    p.fb_layout = std::atoi(argv[2]) ? core_mel::FbLayout::FreqsMels : core_mel::FbLayout::MelsFreqs;
    p.matmul = std::atoi(argv[3]) ? core_mel::MatmulPrecision::Double : core_mel::MatmulPrecision::Float;
    p.n_threads = std::atoi(argv[4]);
    const int repeats = std::atoi(argv[5]);
    if (frames < 1 || p.n_threads < 1 || repeats < 1)
        return 2;
    std::vector<float> pcm((frames - 1) * p.hop_length + p.n_fft);
    for (size_t i = 0; i < pcm.size(); ++i)
        pcm[i] = (float)((int)((i * 167 + 31) % 997) - 498) / 997.f;
    const auto fb = core_mel::build_slaney_fb(16000, p.n_fft, p.n_mels, 0.f, 8000.f, p.fb_layout);
    std::vector<float> window(p.n_fft);
    for (int i = 0; i < p.n_fft; ++i)
        window[i] = 0.5f - 0.5f * std::cos(2.f * 3.14159265358979323846f * i / p.n_fft);
    std::vector<float> expected;
    for (int r = -2; r < repeats; ++r) {
        int actual_frames = 0;
        const auto start = std::chrono::steady_clock::now();
        const auto result = core_mel::compute(pcm.data(), (int)pcm.size(), window.data(), p.n_fft, fb.data(),
                                              p.n_fft / 2 + 1, core_fft::fft_radix2_wrapper, p, actual_frames);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (actual_frames != frames || result.empty())
            throw std::runtime_error("Invalid mel dimensions");
        if (expected.empty())
            expected = result;
        else if (expected != result)
            throw std::runtime_error("Repeated mel output drift");
        if (r >= 0)
            std::fprintf(stderr, "TOTAL_MS %.9f\n", ms);
    }
    double norm2 = 0;
    for (float value : expected) {
        if (!std::isfinite(value))
            throw std::runtime_error("Nonfinite mel");
        norm2 += (double)value * value;
    }
    if (!(norm2 > 0))
        throw std::runtime_error("Zero mel");
    std::fprintf(stderr, "NORM %.17g\n", std::sqrt(norm2));
    FILE* out = std::fopen(argv[6], "wb");
    if (!out)
        return 1;
    const bool ok = std::fwrite(expected.data(), sizeof(float), expected.size(), out) == expected.size();
    return std::fclose(out) == 0 && ok ? 0 : 1;
}
