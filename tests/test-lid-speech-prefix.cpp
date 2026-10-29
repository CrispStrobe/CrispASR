#include "crispasr_lid.h"
#include "core/lid_recording.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

TEST_CASE("LID speech prefix compacts VAD slices", "[unit][lid]") {
    std::vector<float> pcm(100, 0.0f);
    std::fill(pcm.begin() + 10, pcm.begin() + 20, 1.0f);
    std::fill(pcm.begin() + 40, pcm.begin() + 50, 2.0f);
    const std::vector<crispasr_audio_slice> slices = {{10, 20, 0, 0}, {40, 50, 0, 0}};

    const auto got = crispasr_lid_speech_prefix(pcm, slices);
    REQUIRE(got.size() == 10 + 1600 + 10);
    CHECK(std::all_of(got.begin(), got.begin() + 10, [](float v) { return v == 1.0f; }));
    CHECK(std::all_of(got.begin() + 10, got.begin() + 1610, [](float v) { return v == 0.0f; }));
    CHECK(std::all_of(got.begin() + 1610, got.end(), [](float v) { return v == 2.0f; }));
}

TEST_CASE("LID speech prefix clamps slices and caps at 15 seconds", "[unit][lid]") {
    std::vector<float> pcm(16 * 16000, 3.0f);
    const std::vector<crispasr_audio_slice> slices = {{-10, 16 * 16000 + 10, 0, 0}};

    const auto got = crispasr_lid_speech_prefix(pcm, slices);
    REQUIRE(got.size() == 15 * 16000);
    CHECK(std::all_of(got.begin(), got.end(), [](float v) { return v == 3.0f; }));
}

TEST_CASE("LID speech prefix ignores empty slices", "[unit][lid]") {
    const std::vector<float> pcm(10, 1.0f);
    const std::vector<crispasr_audio_slice> slices = {{4, 4, 0, 0}, {20, 30, 0, 0}};
    CHECK(crispasr_lid_speech_prefix(pcm, slices).empty());
}

TEST_CASE("recording-wide LID visits the late speech and weights a short tail by duration", "[unit][lid]") {
    constexpr int sr = 16000;
    std::vector<float> pcm(35 * sr, 0.25f);
    std::vector<int> starts;
    core_lid::WindowResult got;
    const bool ok = core_lid::detect_recording(
        pcm.data(), (int)pcm.size(),
        [&](const float* window, int count, core_lid::WindowResult& vote) {
            const int start = (int)(window - pcm.data());
            starts.push_back(start);
            vote.lang_code = start == 0 ? "en" : "de";
            vote.confidence = start == 0 ? 0.99f : 0.4f;
            vote.source = "whisper";
            CHECK(count == (start == 30 * sr ? 5 : 15) * sr);
            return true;
        },
        got);
    REQUIRE(ok);
    const std::vector<int> expected_starts = {0, 15 * sr, 30 * sr};
    CHECK(starts == expected_starts);
    CHECK(got.lang_code == "de");
    CHECK(got.confidence == Catch::Approx(0.4f * 20.0f / 35.0f));
    CHECK(got.source == "whisper");
}

TEST_CASE("recording-wide LID ignores digital silence and fails on an unscored speech window", "[unit][lid]") {
    constexpr int sr = 16000;
    std::vector<float> pcm(31 * sr, 0.0f);
    std::fill(pcm.begin() + 15 * sr, pcm.end(), 0.2f);
    int calls = 0;
    core_lid::WindowResult got;
    const bool ok = core_lid::detect_recording(
        pcm.data(), (int)pcm.size(),
        [&](const float* window, int count, core_lid::WindowResult& vote) {
            ++calls;
            CHECK(count == 15 * sr);
            CHECK(window - pcm.data() == (calls == 1 ? 15 : 16) * sr);
            vote.lang_code = "de";
            vote.confidence = 0.7f;
            vote.source = "silero";
            return true;
        },
        got);
    REQUIRE(ok);
    CHECK(calls == 2);
    CHECK(got.lang_code == "de");
    CHECK(got.confidence == Catch::Approx(0.7f));

    calls = 0;
    CHECK_FALSE(core_lid::detect_recording(
        pcm.data(), (int)pcm.size(),
        [&](const float*, int, core_lid::WindowResult& vote) {
            ++calls;
            vote.lang_code = "de";
            vote.confidence = 0.7f;
            vote.source = "silero";
            return calls != 2;
        },
        got));
    CHECK(calls == 2);
}
