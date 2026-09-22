// test-mel-blas-parity.cpp — guard against a BLAS that computes the wrong
// mel projection.
//
// WHY THIS EXISTS. `core_mel::compute` projects the power spectrum through the
// filterbank with `cblas_sgemm` when HAVE_BLAS is set (§176f), and that call is
// shared by 40+ backends. On a host where CMake's find_package(BLAS) selects
// Debian's THREADED MKL, `libmkl_intel_thread` is linked into a process that
// already carries libgomp — two OpenMP runtimes — and the sgemm comes back with
// its upper output columns MULTIPLIED BY THE THREAD COUNT: exactly ln(4) of
// error in the log domain at four threads, ln(3) at three, with the lower
// columns correct. Linked against OpenBLAS the same object file is exact.
//
// Nothing raises. The spectrum simply has the wrong shape above a few kHz and
// every model reading it scores worse, which reads as "the model is weak"
// rather than as a bug. It was found by accident while validating the Onsets &
// Frames port against its ONNX export, after the STFT, the filterbank, the
// window and the PCM had each been checked exact and the same `mel.cpp.o`
// still disagreed with itself between two binaries.
//
// THE GUARD. `MatmulPrecision::Double` takes `core_mel`'s own scalar
// accumulation path and never touches BLAS; `MatmulPrecision::Float` takes the
// sgemm when one is linked. Same audio, same window, same filterbank, so the
// two must agree to float rounding. The tolerance below is 1e-3 relative —
// three orders of magnitude tighter than the 3x/4x this is built to catch, per
// crispasr-crispembed-dev.md rule 2c ("a tolerance WIDER than the defect is not
// a test").
//
// A failure here is a TOOLCHAIN fault, not a code fault: link OpenBLAS instead
// of threaded MKL, or build with -DCRISPASR_MEL_BLAS=OFF.

#include <catch2/catch_test_macros.hpp>

#include "core/fft.h"
#include "core/mel.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

void fft_r2c(const float* in, int N, float* out) {
    std::vector<float> re(N), im(N, 0.0f);
    std::memcpy(re.data(), in, (size_t)N * sizeof(float));
    core_fft::fft_radix2_inplace(re.data(), im.data(), N);
    for (int i = 0; i < N; i++) {
        out[2 * i + 0] = re[i];
        out[2 * i + 1] = im[i];
    }
}

// Broadband, deterministic, and non-trivially shaped: a chirp plus a little
// pseudo-random noise, so every filterbank row sees energy and a per-row scale
// error cannot hide in a silent band.
std::vector<float> test_signal(int n) {
    std::vector<float> x((size_t)n);
    unsigned s = 12345u;
    for (int i = 0; i < n; i++) {
        const double t = (double)i / 16000.0;
        s = s * 1103515245u + 12345u;
        const double noise = ((double)((s >> 16) & 0x7fff) / 32768.0 - 0.5) * 0.05;
        x[(size_t)i] = (float)(0.4 * std::sin(2.0 * M_PI * (200.0 + 1500.0 * t) * t) + noise);
    }
    return x;
}

} // namespace

TEST_CASE("core_mel BLAS projection matches the scalar projection", "[mel][unit]") {
    const int n_fft = 2048;
    const int n_freqs = n_fft / 2 + 1;
    // 229 mels is the widest filterbank in the tree (piano-transcription) and
    // therefore the most output columns for a column-partitioned sgemm to get
    // wrong.
    const int n_mels = 229;

    std::vector<float> window((size_t)n_fft);
    for (int i = 0; i < n_fft; i++)
        window[(size_t)i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)i / (float)n_fft);

    const auto fb = core_mel::build_slaney_fb(16000, n_fft, n_mels, 30.0f, 8000.0f, core_mel::FbLayout::MelsFreqs);
    REQUIRE((int)fb.size() == n_mels * n_freqs);

    const auto pcm = test_signal(16000 * 4);

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = 512;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::None; // compare the raw projection
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = 0.0f;
    p.spec_kind = core_mel::SpecKind::Magnitude;
    p.norm = core_mel::Normalization::None;
    p.layout = core_mel::Layout::TimeMels;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.center_pad = true;
    p.center_pad_reflect = true;

    int T_blas = 0, T_scalar = 0;
    p.matmul = core_mel::MatmulPrecision::Float; // cblas_sgemm when linked
    const auto blas =
        core_mel::compute(pcm.data(), (int)pcm.size(), window.data(), n_fft, fb.data(), n_freqs, fft_r2c, p, T_blas);
    p.matmul = core_mel::MatmulPrecision::Double; // core_mel's own scalar loop
    const auto scalar =
        core_mel::compute(pcm.data(), (int)pcm.size(), window.data(), n_fft, fb.data(), n_freqs, fft_r2c, p, T_scalar);

    REQUIRE(T_blas == T_scalar);
    REQUIRE(T_blas > 0);
    REQUIRE(blas.size() == scalar.size());

    double worst = 0.0;
    int worst_mel = -1;
    for (int t = 0; t < T_blas; t++) {
        for (int m = 0; m < n_mels; m++) {
            const size_t i = (size_t)t * n_mels + m;
            const double denom = std::max(1e-6, (double)std::fabs(scalar[i]));
            const double rel = std::fabs((double)blas[i] - (double)scalar[i]) / denom;
            if (rel > worst) {
                worst = rel;
                worst_mel = m;
            }
        }
    }
    INFO("worst relative difference " << worst << " at mel bin " << worst_mel
                                      << " — a ratio near 2, 3 or 4 means the linked BLAS is returning "
                                         "thread-count-scaled columns (threaded MKL alongside libgomp); "
                                         "link OpenBLAS or build with -DCRISPASR_MEL_BLAS=OFF");
    CHECK(worst < 1e-3);
}
