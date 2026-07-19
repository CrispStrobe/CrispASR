#include "core/audio_resample.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

namespace {

float interior_mean(const std::vector<float>& samples) {
    constexpr size_t edge = 128;
    REQUIRE(samples.size() > 2 * edge);

    double sum = 0.0;
    for (size_t i = edge; i < samples.size() - edge; ++i) {
        sum += samples[i];
    }
    return static_cast<float>(sum / static_cast<double>(samples.size() - 2 * edge));
}

} // namespace

TEST_CASE("polyphase resampling preserves DC gain", "[unit][audio-resample]") {
    constexpr float level = 0.25f;
    const std::vector<float> input(48000, level);

    SECTION("48 kHz to 24 kHz") {
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), 48000, 24000);
        REQUIRE(output.size() == 24000);
        REQUIRE(interior_mean(output) == Catch::Approx(level).margin(5e-4f));
    }

    SECTION("24 kHz to 16 kHz") {
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), 24000, 16000);
        REQUIRE(output.size() == 32000);
        REQUIRE(interior_mean(output) == Catch::Approx(level).margin(5e-4f));
    }

    SECTION("24 kHz to 48 kHz") {
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), 24000, 48000);
        REQUIRE(output.size() == 96000);
        REQUIRE(interior_mean(output) == Catch::Approx(level).margin(5e-4f));
    }
}
