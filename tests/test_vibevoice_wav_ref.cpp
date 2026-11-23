// test_vibevoice_wav_ref.cpp - VibeVoice voice-reference WAV regressions.

#include "vibevoice_wav_ref.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

void append_u16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back((uint8_t)(v & 0xff));
    out.push_back((uint8_t)((v >> 8) & 0xff));
}

void append_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back((uint8_t)(v & 0xff));
    out.push_back((uint8_t)((v >> 8) & 0xff));
    out.push_back((uint8_t)((v >> 16) & 0xff));
    out.push_back((uint8_t)((v >> 24) & 0xff));
}

void append_fourcc(std::vector<uint8_t>& out, const char* s) {
    out.insert(out.end(), s, s + 4);
}

void append_chunk(std::vector<uint8_t>& out, const char* id, const std::vector<uint8_t>& payload) {
    append_fourcc(out, id);
    append_u32(out, (uint32_t)payload.size());
    out.insert(out.end(), payload.begin(), payload.end());
    if (payload.size() & 1u)
        out.push_back(0);
}

std::vector<uint8_t> make_wav(const std::vector<int16_t>& samples, uint16_t channels = 1, uint16_t bits_per_sample = 16,
                              uint16_t audio_format = 1, bool add_list_chunk = false) {
    std::vector<uint8_t> fmt;
    append_u16(fmt, audio_format);
    append_u16(fmt, channels);
    append_u32(fmt, 24000);
    append_u32(fmt, 24000u * channels * bits_per_sample / 8u);
    append_u16(fmt, (uint16_t)(channels * bits_per_sample / 8u));
    append_u16(fmt, bits_per_sample);

    std::vector<uint8_t> data;
    for (int16_t sample : samples)
        append_u16(data, (uint16_t)sample);

    std::vector<uint8_t> chunks;
    append_chunk(chunks, "fmt ", fmt);
    if (add_list_chunk) {
        std::vector<uint8_t> list = {'I', 'N', 'F', 'O', 'I', 'S', 'F', 'T', 3, 0, 0, 0, 'f', 'f', 'm'};
        append_chunk(chunks, "LIST", list);
    }
    append_chunk(chunks, "data", data);

    std::vector<uint8_t> out;
    append_fourcc(out, "RIFF");
    append_u32(out, (uint32_t)(4 + chunks.size()));
    append_fourcc(out, "WAVE");
    out.insert(out.end(), chunks.begin(), chunks.end());
    return out;
}

} // namespace

TEST_CASE("VibeVoice voice reference parses mono PCM16 WAV", "[unit][vibevoice]") {
    const std::vector<uint8_t> wav = make_wav({0, 16384, -32768, 32767});
    std::vector<float> pcm;

    REQUIRE(vibevoice_parse_mono_pcm16_wav(wav.data(), wav.size(), pcm));
    REQUIRE(pcm.size() == 4);
    REQUIRE(pcm[0] == 0.0f);
    REQUIRE(pcm[1] == Catch::Approx(0.5f));
    REQUIRE(pcm[2] == Catch::Approx(-1.0f));
    REQUIRE(pcm[3] == Catch::Approx(32767.0f / 32768.0f));
}

TEST_CASE("VibeVoice voice reference accepts metadata chunks before data", "[unit][vibevoice]") {
    const std::vector<uint8_t> wav = make_wav({100, -100}, 1, 16, 1, true);
    std::vector<float> pcm;

    REQUIRE(vibevoice_parse_mono_pcm16_wav(wav.data(), wav.size(), pcm));
    REQUIRE(pcm.size() == 2);
    REQUIRE(pcm[0] == Catch::Approx(100.0f / 32768.0f));
    REQUIRE(pcm[1] == Catch::Approx(-100.0f / 32768.0f));
}

TEST_CASE("VibeVoice voice reference rejects unsupported WAV formats", "[unit][vibevoice]") {
    std::vector<float> pcm;
    const std::vector<uint8_t> adpcm = make_wav({0}, 1, 16, 2);
    const std::vector<uint8_t> float16 = make_wav({0}, 1, 16, 3); // float must be 32/64-bit
    std::vector<uint8_t> not_wav = {'N', 'O', 'P', 'E'};

    REQUIRE_FALSE(vibevoice_parse_wav_mono(adpcm.data(), adpcm.size(), pcm));
    REQUIRE_FALSE(vibevoice_parse_wav_mono(float16.data(), float16.size(), pcm));
    REQUIRE_FALSE(vibevoice_parse_wav_mono(not_wav.data(), not_wav.size(), pcm));
}

// #589: a stereo reference was rejected and the CLI silently fell back to the
// generic voice. Upstream (librosa mono=True) averages the channels.
TEST_CASE("VibeVoice voice reference downmixes stereo to mono", "[unit][vibevoice]") {
    const std::vector<uint8_t> wav = make_wav({16384, 0, -16384, -16384, 1000, 3000}, 2);
    std::vector<float> pcm;
    uint32_t sr = 0;

    REQUIRE(vibevoice_parse_wav_mono(wav.data(), wav.size(), pcm, &sr));
    REQUIRE(sr == 24000);
    REQUIRE(pcm.size() == 3);
    REQUIRE(pcm[0] == Catch::Approx(0.25f));
    REQUIRE(pcm[1] == Catch::Approx(-0.5f));
    REQUIRE(pcm[2] == Catch::Approx(2000.0f / 32768.0f));
}

TEST_CASE("VibeVoice voice reference reads 8/24/32-bit PCM and float", "[unit][vibevoice]") {
    auto wav_bytes = [](const std::vector<uint8_t>& payload, uint16_t channels, uint16_t bits, uint16_t fmt_code,
                        bool extensible) {
        std::vector<uint8_t> fmt;
        append_u16(fmt, extensible ? 0xFFFE : fmt_code);
        append_u16(fmt, channels);
        append_u32(fmt, 16000);
        append_u32(fmt, 16000u * channels * bits / 8u);
        append_u16(fmt, (uint16_t)(channels * bits / 8u));
        append_u16(fmt, bits);
        if (extensible) {
            append_u16(fmt, 22);       // cbSize
            append_u16(fmt, bits);     // valid bits
            append_u32(fmt, 0x3);      // channel mask
            append_u16(fmt, fmt_code); // SubFormat GUID, first two bytes
            const uint8_t rest[14] = {0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80,
                                      0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
            fmt.insert(fmt.end(), rest, rest + 14);
        }
        std::vector<uint8_t> chunks;
        append_chunk(chunks, "fmt ", fmt);
        append_chunk(chunks, "data", payload);
        std::vector<uint8_t> out;
        append_fourcc(out, "RIFF");
        append_u32(out, (uint32_t)(4 + chunks.size()));
        append_fourcc(out, "WAVE");
        out.insert(out.end(), chunks.begin(), chunks.end());
        return out;
    };
    std::vector<float> pcm;
    uint32_t sr = 0;

    // 8-bit unsigned: 192 -> +0.5
    auto w8 = wav_bytes({192}, 1, 8, 1, false);
    REQUIRE(vibevoice_parse_wav_mono(w8.data(), w8.size(), pcm, &sr));
    REQUIRE(sr == 16000);
    REQUIRE(pcm.size() == 1);
    REQUIRE(pcm[0] == Catch::Approx(0.5f));

    // 24-bit: 0x400000 -> +0.5, 0xC00000 -> -0.5
    auto w24 = wav_bytes({0x00, 0x00, 0x40, 0x00, 0x00, 0xC0}, 1, 24, 1, false);
    REQUIRE(vibevoice_parse_wav_mono(w24.data(), w24.size(), pcm));
    REQUIRE(pcm.size() == 2);
    REQUIRE(pcm[0] == Catch::Approx(0.5f));
    REQUIRE(pcm[1] == Catch::Approx(-0.5f));

    // 32-bit float stereo in WAVE_FORMAT_EXTENSIBLE: (0.5 + -0.25) / 2
    std::vector<uint8_t> f32;
    for (float v : {0.5f, -0.25f}) {
        uint8_t b[4];
        std::memcpy(b, &v, 4);
        f32.insert(f32.end(), b, b + 4);
    }
    auto wf = wav_bytes(f32, 2, 32, 3, true);
    REQUIRE(vibevoice_parse_wav_mono(wf.data(), wf.size(), pcm));
    REQUIRE(pcm.size() == 1);
    REQUIRE(pcm[0] == Catch::Approx(0.125f));

    // 32-bit integer PCM: 0x40000000 -> +0.5
    auto w32 = wav_bytes({0x00, 0x00, 0x00, 0x40}, 1, 32, 1, false);
    REQUIRE(vibevoice_parse_wav_mono(w32.data(), w32.size(), pcm));
    REQUIRE(pcm[0] == Catch::Approx(0.5f));
}

TEST_CASE("VibeVoice voice reference handles extremely short WAV", "[unit][vibevoice]") {
    const std::vector<uint8_t> wav = make_wav({42});
    std::vector<float> pcm;

    REQUIRE(vibevoice_parse_mono_pcm16_wav(wav.data(), wav.size(), pcm));
    REQUIRE(pcm.size() == 1);
    REQUIRE(pcm[0] == Catch::Approx(42.0f / 32768.0f));
}

TEST_CASE("VibeVoice voice reference handles high amplitude (clipping)", "[unit][vibevoice]") {
    // 32767 is max positive i16
    const std::vector<uint8_t> wav = make_wav({32767, -32768});
    std::vector<float> pcm;

    REQUIRE(vibevoice_parse_mono_pcm16_wav(wav.data(), wav.size(), pcm));
    REQUIRE(pcm.size() == 2);
    REQUIRE(std::abs(pcm[0]) <= 1.0f);
    REQUIRE(std::abs(pcm[1]) <= 1.0f);
}

TEST_CASE("VibeVoice voice reference rejects empty data", "[unit][vibevoice]") {
    std::vector<float> pcm;
    REQUIRE_FALSE(vibevoice_parse_mono_pcm16_wav(nullptr, 0, pcm));
}

TEST_CASE("VibeVoice voice reference normalization preserves waveform shape", "[unit][vibevoice]") {
    std::vector<float> pcm = {1.0f, -0.5f, 0.25f};

    vibevoice_normalize_ref_pcm(pcm, 0.0f);

    const float max_abs = std::max({std::fabs(pcm[0]), std::fabs(pcm[1]), std::fabs(pcm[2])});
    REQUIRE(max_abs <= 1.0f);
    REQUIRE(pcm[0] / pcm[1] == Catch::Approx(-2.0f).epsilon(0.001));
    REQUIRE(pcm[1] / pcm[2] == Catch::Approx(-2.0f).epsilon(0.001));
}

TEST_CASE("VibeVoice voice reference normalization leaves silence finite", "[unit][vibevoice]") {
    std::vector<float> pcm = {0.0f, 0.0f, 0.0f};

    vibevoice_normalize_ref_pcm(pcm);

    REQUIRE(pcm.size() == 3);
    REQUIRE(std::all_of(pcm.begin(), pcm.end(), [](float v) { return v == 0.0f; }));
}
