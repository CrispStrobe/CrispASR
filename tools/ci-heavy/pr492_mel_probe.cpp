// Serial/parallel projection A/B, including the T=64 branch boundary.
#include "core/fft.h"
#include "core/mel.h"
#include <cstdio>
#include <stdexcept>
#include <vector>

static void fft(const float* in, int n, float* out) {
    std::vector<float> re(in, in + n), im(n, 0.f);
    core_fft::fft_radix2_inplace(re.data(), im.data(), n);
    for (int i = 0; i < n; ++i) {
        out[2 * i] = re[i];
        out[2 * i + 1] = im[i];
    }
}
int main() {
    int cases = 0;
    for (int frames : {63, 64, 65, 300}) {
        for (auto layout : {core_mel::FbLayout::MelsFreqs, core_mel::FbLayout::FreqsMels}) {
            for (auto precision : {core_mel::MatmulPrecision::Float, core_mel::MatmulPrecision::Double}) {
                core_mel::Params p;
                p.n_fft = p.win_length = 64;
                p.hop_length = 16;
                p.n_mels = 17;
                p.center_pad = false;
                p.log_base = core_mel::LogBase::None;
                p.norm = core_mel::Normalization::None;
                p.layout = core_mel::Layout::TimeMels;
                p.fb_layout = layout;
                p.matmul = precision;
                std::vector<float> pcm((frames - 1) * p.hop_length + p.n_fft);
                for (size_t i = 0; i < pcm.size(); ++i)
                    pcm[i] = (float)((int)((i * 167 + 31) % 997) - 498) / 997.f;
                const auto fb = core_mel::build_slaney_fb(16000, 64, 17, 0.f, 8000.f, layout);
                std::vector<float> window(p.n_fft, 1.f);
                int serial_t = 0, parallel_t = 0;
                p.n_threads = 1;
                const auto serial =
                    core_mel::compute(pcm.data(), pcm.size(), window.data(), 64, fb.data(), 33, fft, p, serial_t);
                p.n_threads = 4;
                const auto parallel =
                    core_mel::compute(pcm.data(), pcm.size(), window.data(), 64, fb.data(), 33, fft, p, parallel_t);
                if (serial_t != frames || parallel_t != frames || serial != parallel)
                    throw std::runtime_error("mel projection changed with thread budget");
                if (std::fwrite(parallel.data(), sizeof(float), parallel.size(), stdout) != parallel.size())
                    return 1;
                ++cases;
            }
        }
    }
    std::fprintf(stderr, "MEL_PROJECTION_PASS: %d cases, exact at 1/4 threads\n", cases);
}
