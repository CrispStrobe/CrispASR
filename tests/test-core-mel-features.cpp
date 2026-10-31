// test-core-mel-features.cpp — verification of core_mel normalization, pre-emphasis,
// layout, and multithreaded bit-exact parity across feature configurations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/fft.h"
#include "core/mel.h"

#include <cmath>
#include <vector>

namespace {

void test_fft_r2c(const float* in, int N, float* out) {
    core_fft::fft_radix2_wrapper(in, N, out);
}

std::vector<float> generate_synthetic_audio(int n_samples, int sample_rate = 16000) {
    std::vector<float> audio((size_t)n_samples);
    for (int i = 0; i < n_samples; i++) {
        const double t = (double)i / sample_rate;
        audio[(size_t)i] = 0.6f * std::sin(2.0 * M_PI * 300.0 * t) +
                           0.3f * std::sin(2.0 * M_PI * 1200.0 * t) +
                           0.1f * std::sin(2.0 * M_PI * 3400.0 * t);
    }
    return audio;
}

std::vector<float> generate_window(int n_fft) {
    std::vector<float> win((size_t)n_fft);
    for (int i = 0; i < n_fft; i++) {
        win[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)(n_fft - 1)));
    }
    return win;
}

} // namespace

TEST_CASE("core_mel PerFeatureZ normalization with pre-emphasis and reflect-padding", "[mel][unit]") {
    const int sr = 16000;
    const int n_fft = 512;
    const int n_mels = 80;
    const int n_freqs = n_fft / 2 + 1;
    const int n_samples = sr * 4; // 4 seconds of audio

    const auto audio = generate_synthetic_audio(n_samples, sr);
    const auto win = generate_window(n_fft);
    const auto fb = core_mel::build_slaney_fb(sr, n_fft, n_mels, 0.0f, 8000.0f, core_mel::FbLayout::MelsFreqs);

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = 160;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.log_eps = 1e-5f;
    p.log_guard = core_mel::LogGuard::AddEpsilon;
    p.spec_kind = core_mel::SpecKind::Power;
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.center_pad = true;
    p.center_pad_reflect = true;
    p.preemph = 0.97f;

    int T_single = 0;
    p.n_threads = 1;
    const auto mel_single = core_mel::compute(audio.data(), (int)audio.size(), win.data(), n_fft, fb.data(), n_freqs,
                                              test_fft_r2c, p, T_single);

    int T_multi = 0;
    p.n_threads = 8;
    const auto mel_multi = core_mel::compute(audio.data(), (int)audio.size(), win.data(), n_fft, fb.data(), n_freqs,
                                             test_fft_r2c, p, T_multi);

    REQUIRE(T_single > 0);
    REQUIRE(T_single == T_multi);
    REQUIRE(mel_single.size() == mel_multi.size());

    // Verify bit-identical results across all elements
    size_t diff_count = 0;
    for (size_t i = 0; i < mel_single.size(); i++) {
        if (mel_single[i] != mel_multi[i]) {
            diff_count++;
        }
    }
    REQUIRE(diff_count == 0);

    // Verify PerFeatureZ properties: per mel band mean should be close to 0 and std close to 1 / (std + 1e-5)
    for (int m = 0; m < n_mels; m++) {
        double sum = 0.0;
        for (int t = 0; t < T_single; t++) {
            sum += mel_single[(size_t)t * n_mels + m];
        }
        const double mean = sum / T_single;
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0.0, 1e-4));
    }
}

TEST_CASE("core_mel MelsTime transposed layout bit-exact parity", "[mel][unit]") {
    const int sr = 16000;
    const int n_fft = 512;
    const int n_mels = 128;
    const int n_freqs = n_fft / 2 + 1;
    const int n_samples = sr * 3;

    const auto audio = generate_synthetic_audio(n_samples, sr);
    const auto win = generate_window(n_fft);
    const auto fb = core_mel::build_htk_fb(sr, n_fft, n_mels, 0.0f, 8000.0f, core_mel::FbLayout::MelsFreqs);

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = 160;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_eps = 1e-10f;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.norm = core_mel::Normalization::None;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.center_pad = true;
    p.center_pad_reflect = false;

    int T_single = 0;
    p.n_threads = 1;
    const auto mel_single = core_mel::compute(audio.data(), (int)audio.size(), win.data(), n_fft, fb.data(), n_freqs,
                                              test_fft_r2c, p, T_single);

    int T_multi = 0;
    p.n_threads = 8;
    const auto mel_multi = core_mel::compute(audio.data(), (int)audio.size(), win.data(), n_fft, fb.data(), n_freqs,
                                             test_fft_r2c, p, T_multi);

    REQUIRE(T_single > 0);
    REQUIRE(T_single == T_multi);
    REQUIRE(mel_single.size() == mel_multi.size());

    size_t diff_count = 0;
    for (size_t i = 0; i < mel_single.size(); i++) {
        if (mel_single[i] != mel_multi[i]) {
            diff_count++;
        }
    }
    REQUIRE(diff_count == 0);
}

TEST_CASE("core_mel GlobalClipMax normalization bit-exact parity", "[mel][unit]") {
    const int sr = 16000;
    const int n_fft = 400;
    const int n_mels = 80;
    const int n_freqs = n_fft / 2 + 1;
    const int n_samples = sr * 2;

    const auto audio = generate_synthetic_audio(n_samples, sr);
    const auto win = generate_window(n_fft);
    const auto fb = core_mel::build_slaney_fb(sr, n_fft, n_mels, 0.0f, 8000.0f, core_mel::FbLayout::MelsFreqs);

    core_mel::Params p;
    p.n_fft = 512;
    p.hop_length = 160;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = 1e-5f;
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::TimeMels;
    p.center_pad = true;

    int T_single = 0;
    p.n_threads = 1;
    const auto mel_single = core_mel::compute(audio.data(), (int)audio.size(), win.data(), n_fft, fb.data(), n_freqs,
                                              test_fft_r2c, p, T_single);

    int T_multi = 0;
    p.n_threads = 8;
    const auto mel_multi = core_mel::compute(audio.data(), (int)audio.size(), win.data(), n_fft, fb.data(), n_freqs,
                                             test_fft_r2c, p, T_multi);

    REQUIRE(T_single > 0);
    REQUIRE(T_single == T_multi);
    REQUIRE(mel_single.size() == mel_multi.size());

    size_t diff_count = 0;
    for (size_t i = 0; i < mel_single.size(); i++) {
        if (mel_single[i] != mel_multi[i]) {
            diff_count++;
        }
    }
    REQUIRE(diff_count == 0);
}
