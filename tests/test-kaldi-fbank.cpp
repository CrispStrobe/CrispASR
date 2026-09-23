// test-kaldi-fbank.cpp — core_kaldi::compute_fbank against kaldi-native-fbank.
//
// The snip_edges=false + mel-domain path (X-ASR / sherpa-onnx, #436) is pinned
// to golden frames from kaldi-native-fbank on a deterministic signal: frame 0
// and the last frame exercise the mirrored edges, the middle one the plain
// window. Also pinned: the frame-count rules, and that the per-thread filter
// cache never hands one configuration another's filters.

#include "core/kaldi_fbank.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// kaldi-native-fbank 1.22.3: snip_edges=false, 80 bins, 20..-400 Hz, dither 0
static const int kKnfFrames = 25;
static const float kKnfFrame0[80] = {
    -5.547179f, -4.917483f, -4.656730f, -4.365270f, -3.922342f, -3.466322f, -2.980455f, -2.484073f, -2.226295f,
    -1.598216f, -0.923988f, -0.250407f, 0.897814f,  2.236650f,  2.657355f,  2.822258f,  1.719635f,  0.309735f,
    -0.281769f, -0.889821f, -1.121744f, -1.598309f, -1.849134f, -2.023794f, -2.438363f, -2.435510f, -2.544171f,
    -2.869947f, -3.115895f, -3.234181f, -3.505492f, -3.782467f, -3.744403f, -4.285954f, -4.465219f, -4.758625f,
    -5.161709f, -4.866463f, -5.957814f, -6.214346f, -4.973634f, -5.406401f, -4.507827f, -4.899536f, -4.435064f,
    -3.226037f, -3.145459f, -2.362637f, -2.073056f, -1.174091f, -0.539651f, 0.296134f,  2.308714f,  4.994944f,
    4.368043f,  1.260890f,  0.357985f,  -0.288577f, -0.634257f, -0.896542f, -1.402953f, -1.601463f, -1.788298f,
    -1.854062f, -2.171356f, -2.585665f, -2.648629f, -3.110073f, -2.402375f, -2.355141f, -2.888559f, -3.037684f,
    -3.176050f, -2.660958f, -3.450950f, -3.573622f, -4.512594f, -3.664487f, -2.245971f, -3.298790f};
static const float kKnfFrame12[80] = {
    -12.820503f, -10.932946f, -13.005976f, -12.118986f, -11.770904f, -10.951439f, -8.966280f, -8.809894f, -9.323437f,
    -8.590181f,  -5.781736f,  -3.925746f,  -2.172375f,  2.080165f,   3.326128f,   2.893050f,  0.165839f,  -3.569039f,
    -6.500548f,  -8.551346f,  -7.403048f,  -10.023656f, -11.521047f, -9.522612f,  -8.726211f, -8.330978f, -8.460464f,
    -8.367172f,  -8.537580f,  -7.195136f,  -7.268104f,  -7.552956f,  -6.727006f,  -6.202927f, -7.625761f, -7.250937f,
    -5.884592f,  -6.053092f,  -6.455292f,  -7.226170f,  -6.338294f,  -5.748717f,  -6.076550f, -5.946497f, -6.784163f,
    -7.307064f,  -5.825921f,  -5.071545f,  -5.930933f,  -5.558425f,  -4.364944f,  -4.670566f, 1.145940f,  5.206970f,
    4.099787f,   -3.635526f,  -4.170474f,  -3.792204f,  -4.463105f,  -4.535103f,  -5.148811f, -5.214955f, -4.521631f,
    -4.323409f,  -4.705513f,  -4.167836f,  -3.981893f,  -4.635414f,  -4.547337f,  -3.965861f, -4.131447f, -3.937719f,
    -3.743978f,  -3.364270f,  -2.956718f,  -3.580485f,  -3.794633f,  -3.284913f,  -3.529481f, -3.216855f};
static const float kKnfFrame24[80] = {
    -5.437695f, -4.370836f, -4.156723f, -4.022630f, -3.684211f, -3.293596f, -2.834610f, -2.357821f, -2.090793f,
    -1.518558f, -0.909242f, -0.194230f, 0.952625f,  2.379169f,  2.680315f,  2.677458f,  1.670316f,  0.277622f,
    -0.260260f, -0.869111f, -1.120665f, -1.589111f, -1.845478f, -2.002711f, -2.339823f, -2.463748f, -2.655381f,
    -2.738235f, -2.992474f, -3.120164f, -3.459073f, -3.589802f, -3.591704f, -3.926597f, -4.113853f, -5.030826f,
    -4.514957f, -5.049874f, -7.002467f, -7.495137f, -6.567356f, -6.008674f, -4.899625f, -4.728527f, -3.790187f,
    -3.117389f, -2.839291f, -2.171826f, -2.015586f, -1.438313f, -0.644208f, 0.386521f,  2.598840f,  5.136604f,
    3.915357f,  1.242058f,  0.213872f,  -0.482078f, -0.947779f, -1.437075f, -1.237684f, -1.746325f, -1.893498f,
    -2.352452f, -2.491088f, -2.757187f, -2.497176f, -3.313998f, -3.322191f, -2.449777f, -2.494498f, -2.291312f,
    -2.900843f, -3.200339f, -3.030466f, -2.530895f, -3.627817f, -3.466594f, -3.824167f, -3.442827f};

std::vector<float> signal_4000() {
    std::vector<float> a(4000);
    uint32_t x = 12345;
    for (int i = 0; i < 4000; i++) {
        x = 1103515245u * x + 12345u;
        const double noise = (double)(x >> 8) / (double)(1u << 24) - 0.5;
        const double t = i / 16000.0;
        a[(size_t)i] = (float)(0.3 * std::sin(2 * kPi * 440 * t) + 0.1 * std::sin(2 * kPi * 3000 * t) + 0.01 * noise);
    }
    return a;
}

core_kaldi::FbankParams sherpa_params() {
    core_kaldi::FbankParams p;
    p.high_freq = -400.0f;
    p.snip_edges = false;
    p.mel_domain_triangles = true;
    return p;
}

} // namespace

TEST_CASE("snip_edges=false + mel-domain matches kaldi-native-fbank", "[kaldi_fbank]") {
    const auto a = signal_4000();
    int T = 0;
    const auto fb = core_kaldi::compute_fbank(a.data(), (int)a.size(), sherpa_params(), T);
    REQUIRE(T == kKnfFrames);
    const float* want[3] = {kKnfFrame0, kKnfFrame12, kKnfFrame24};
    const int rows[3] = {0, 12, 24};
    for (int r = 0; r < 3; r++) {
        for (int m = 0; m < 80; m++) {
            INFO("frame " << rows[r] << " bin " << m);
            CHECK(std::fabs(fb[(size_t)rows[r] * 80 + m] - want[r][m]) < 2e-3f);
        }
    }
}

TEST_CASE("frame counts follow kaldi NumFrames", "[kaldi_fbank]") {
    std::vector<float> a(5000, 0.01f);
    for (int n : {1, 79, 80, 239, 240, 400, 401, 4999}) {
        int T = 0;
        core_kaldi::FbankParams p = sherpa_params();
        core_kaldi::compute_fbank(a.data(), n, p, T);
        CHECK(T == (n + 80) / 160);
        p.snip_edges = true;
        core_kaldi::compute_fbank(a.data(), n, p, T);
        CHECK(T == (n < 400 ? 0 : (n - 400) / 160 + 1));
    }
}

TEST_CASE("filter cache keys on every bank parameter", "[kaldi_fbank]") {
    const auto a = signal_4000();
    core_kaldi::FbankParams hz; // historic defaults
    int T = 0;
    const auto first = core_kaldi::compute_fbank(a.data(), (int)a.size(), hz, T);
    const auto mel = core_kaldi::compute_fbank(a.data(), (int)a.size(), sherpa_params(), T);
    core_kaldi::FbankParams narrow = hz;
    narrow.high_freq = 4000.0f;
    const auto n4k = core_kaldi::compute_fbank(a.data(), (int)a.size(), narrow, T);
    const auto again = core_kaldi::compute_fbank(a.data(), (int)a.size(), hz, T);
    REQUIRE(first == again);
    CHECK(n4k[40] != first[40]); // same sr / n_fft / n_mels, different band
    (void)mel;
}
