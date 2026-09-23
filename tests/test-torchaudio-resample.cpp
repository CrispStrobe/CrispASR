// core_torchaudio::resample must reproduce torchaudio.functional.resample
// (sinc_interp_hann, width 6, rolloff 0.99). Raon-Speech (#455) resamples
// 16k -> 24k -> 16k before its mel, so a generic resampler would shift the
// reference features.
#include "core/torchaudio_resample.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

TEST_CASE("sinc kernel 16k->24k equals torchaudio's", "[unit][resample]") {
    // torchaudio kernel as baked into src/indextts.cpp (dumped from torchaudio)
    static const float K[3][16] = {
        {0.f, -2.4526139e-06f, 7.3377293e-04f, -2.5844171e-03f, 5.0710216e-03f, -7.5402437e-03f, 9.3416208e-03f,
         9.9000001e-01f, 9.3416208e-03f, -7.5402437e-03f, 5.0710216e-03f, -2.5844171e-03f, 7.3377293e-04f,
         -2.4526139e-06f, 0.f, 0.f},
        {0.f, 0.f, -5.4905243e-04f, 7.9238983e-03f, -2.6932398e-02f, 6.4122014e-02f, -1.4034304e-01f, 4.0603769e-01f,
         8.1582844e-01f, -1.7843980e-01f, 7.6355681e-02f, -3.2585010e-02f, 1.0875782e-02f, -1.6146711e-03f, 0.f, 0.f},
        {0.f, 0.f, 0.f, -1.6146711e-03f, 1.0875782e-02f, -3.2585010e-02f, 7.6355688e-02f, -1.7843981e-01f,
         8.1582838e-01f, 4.0603778e-01f, -1.4034306e-01f, 6.4122021e-02f, -2.6932402e-02f, 7.9239001e-03f,
         -5.4905261e-04f, 0.f},
    };
    const auto k = core_torchaudio::sinc_resample_kernel(16000, 24000);
    REQUIRE(k.orig == 2);
    REQUIRE(k.nw == 3);
    REQUIRE(k.width == 7);
    REQUIRE(k.n_taps() == 16);
    for (int p = 0; p < 3; p++)
        for (int j = 0; j < 16; j++)
            REQUIRE_THAT(k.taps[p * 16 + j], Catch::Matchers::WithinAbs(K[p][j], 5e-7));
}

TEST_CASE("resample output lengths follow ceil(new * len / orig)", "[unit][resample]") {
    std::vector<float> x(1001, 0.25f);
    REQUIRE(core_torchaudio::resample(x.data(), 1001, 16000, 24000).size() == 1502);
    REQUIRE(core_torchaudio::resample(x.data(), 1001, 24000, 16000).size() == 668);
    REQUIRE(core_torchaudio::resample(x.data(), 1001, 16000, 16000).size() == 1001);
    const auto k = core_torchaudio::sinc_resample_kernel(24000, 16000);
    REQUIRE(k.width == 10); // ceil(6 * 3 / 1.98)
}

TEST_CASE("a band-limited tone survives 16k->24k->16k", "[unit][resample]") {
    const int n = 4000;
    std::vector<float> x(n);
    for (int i = 0; i < n; i++)
        x[i] = std::sin(2.0f * 3.14159265f * 440.0f * i / 16000.0f);
    const auto up = core_torchaudio::resample(x.data(), n, 16000, 24000);
    const auto back = core_torchaudio::resample(up.data(), (int)up.size(), 24000, 16000);
    REQUIRE((int)back.size() == n);
    double err = 0.0;
    for (int i = 200; i < n - 200; i++) // away from the zero-padded edges
        err = std::fmax(err, std::fabs(back[i] - x[i]));
    REQUIRE(err < 1e-3);
}
