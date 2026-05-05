// test_server_wav_writer.cpp — unit tests for crispasr_make_wav_int16.
//
// The WAV serializer used by /v1/audio/speech and crispasr_diff. The
// header layout is fixed by RFC 2361 (WAVEFORMATEX); we test every
// field at the byte level so a future "tidy-up" refactor can't
// silently change the wire format that downstream OpenAI clients
// depend on.

#include "crispasr_wav_writer.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

// LE byte readers used by every test below — RIFF is little-endian by
// spec, our serializer writes byte-by-byte regardless of host endianness,
// so we read the same way to keep the test endian-independent.
uint16_t le_u16(const std::string& s, size_t off) {
    return (uint8_t)s[off] | ((uint16_t)(uint8_t)s[off + 1] << 8);
}
uint32_t le_u32(const std::string& s, size_t off) {
    return (uint32_t)(uint8_t)s[off] | ((uint32_t)(uint8_t)s[off + 1] << 8) | ((uint32_t)(uint8_t)s[off + 2] << 16) |
           ((uint32_t)(uint8_t)s[off + 3] << 24);
}
int16_t le_i16(const std::string& s, size_t off) {
    return (int16_t)le_u16(s, off);
}

} // namespace

TEST_CASE("WAV header magic bytes", "[unit][wav]") {
    std::vector<float> samples = {0.0f, 0.5f, -0.5f};
    std::string wav = crispasr_make_wav_int16(samples.data(), (int)samples.size(), 24000);

    REQUIRE(wav.size() >= 44);
    REQUIRE(wav.substr(0, 4) == "RIFF");
    REQUIRE(wav.substr(8, 4) == "WAVE");
    REQUIRE(wav.substr(12, 4) == "fmt ");
    REQUIRE(wav.substr(36, 4) == "data");
}

TEST_CASE("WAV header field values", "[unit][wav]") {
    std::vector<float> samples(100, 0.0f);
    const int rate = 24000;
    std::string wav = crispasr_make_wav_int16(samples.data(), (int)samples.size(), rate);

    // RIFF size = 36 + data_size.
    REQUIRE(le_u32(wav, 4) == 36 + samples.size() * 2);

    // fmt chunk size is 16 for PCM.
    REQUIRE(le_u32(wav, 16) == 16);

    // Format = 1 (PCM).
    REQUIRE(le_u16(wav, 20) == 1);

    // Mono.
    REQUIRE(le_u16(wav, 22) == 1);

    // Sample rate.
    REQUIRE(le_u32(wav, 24) == (uint32_t)rate);

    // Byte rate = rate * channels * bytes-per-sample = rate * 1 * 2.
    REQUIRE(le_u32(wav, 28) == (uint32_t)rate * 2);

    // Block align = channels * bytes-per-sample = 2.
    REQUIRE(le_u16(wav, 32) == 2);

    // Bits per sample.
    REQUIRE(le_u16(wav, 34) == 16);

    // data chunk size.
    REQUIRE(le_u32(wav, 40) == samples.size() * 2);
}

TEST_CASE("WAV body size matches sample count", "[unit][wav]") {
    for (int n : {1, 17, 256, 24000, 100000}) {
        std::vector<float> samples(n, 0.5f);
        std::string wav = crispasr_make_wav_int16(samples.data(), n, 24000);
        REQUIRE(wav.size() == 44 + (size_t)n * 2);
    }
}

TEST_CASE("WAV serializer clamps samples outside [-1, 1]", "[unit][wav]") {
    // Without clamping, +1.5 * 32767 = 49150 wraps as int16. We need
    // the wire bytes to stay at +/- 32767 instead.
    std::vector<float> samples = {0.0f, 1.5f, -1.5f, 2.0f, -2.0f};
    std::string wav = crispasr_make_wav_int16(samples.data(), (int)samples.size(), 24000);

    REQUIRE(le_i16(wav, 44 + 0) == 0);
    REQUIRE(le_i16(wav, 44 + 2) == 32767);
    REQUIRE(le_i16(wav, 44 + 4) == -32767);
    REQUIRE(le_i16(wav, 44 + 6) == 32767);
    REQUIRE(le_i16(wav, 44 + 8) == -32767);
}

TEST_CASE("WAV serializer rounds samples correctly", "[unit][wav]") {
    // 0.5f * 32767 = 16383.5 → lround → 16384 (banker's rounding off,
    // round-half-away-from-zero on POSIX).
    // -0.5f * 32767 = -16383.5 → lround → -16384.
    // Tiny epsilons should round to 0 and 1.
    std::vector<float> samples = {0.5f, -0.5f, 0.00001f, 0.000031f};
    std::string wav = crispasr_make_wav_int16(samples.data(), (int)samples.size(), 24000);

    REQUIRE(le_i16(wav, 44 + 0) == 16384);
    REQUIRE(le_i16(wav, 44 + 2) == -16384);
    REQUIRE(le_i16(wav, 44 + 4) == 0);
    REQUIRE(le_i16(wav, 44 + 6) == 1);
}

TEST_CASE("WAV serializer handles boundary values", "[unit][wav]") {
    std::vector<float> samples = {1.0f, -1.0f};
    std::string wav = crispasr_make_wav_int16(samples.data(), (int)samples.size(), 24000);

    REQUIRE(le_i16(wav, 44 + 0) == 32767);
    REQUIRE(le_i16(wav, 44 + 2) == -32767);
}

TEST_CASE("WAV serializer accepts varied sample rates", "[unit][wav]") {
    std::vector<float> samples = {0.0f};
    for (int rate : {8000, 16000, 22050, 24000, 44100, 48000, 96000}) {
        std::string wav = crispasr_make_wav_int16(samples.data(), 1, rate);
        REQUIRE(le_u32(wav, 24) == (uint32_t)rate);
        REQUIRE(le_u32(wav, 28) == (uint32_t)rate * 2);
    }
}

TEST_CASE("WAV serializer handles empty input", "[unit][wav]") {
    // n_samples = 0 should produce a header-only WAV (44 bytes), not
    // crash and not undersize the buffer.
    std::string wav = crispasr_make_wav_int16(nullptr, 0, 24000);
    REQUIRE(wav.size() == 44);
    REQUIRE(wav.substr(0, 4) == "RIFF");
    REQUIRE(le_u32(wav, 4) == 36); // 36 + 0
    REQUIRE(le_u32(wav, 40) == 0); // data size

    // Negative sample count: be defensive — treat as zero, do not
    // dereference the pointer (which may be nullptr).
    std::string wav_neg = crispasr_make_wav_int16(nullptr, -5, 24000);
    REQUIRE(wav_neg.size() == 44);
    REQUIRE(le_u32(wav_neg, 40) == 0);
}

TEST_CASE("WAV total size in RIFF header matches actual byte count", "[unit][wav]") {
    // The "RIFF size" field is the file size minus 8 (RIFF header + size field
    // itself). Verify that invariant for several lengths.
    for (int n : {0, 1, 100, 24000}) {
        std::vector<float> samples(n, 0.0f);
        std::string wav = crispasr_make_wav_int16(samples.data(), n, 24000);
        REQUIRE(le_u32(wav, 4) + 8 == wav.size());
    }
}

// ──────────────────────────────────────────────────────────────────────────
// crispasr_make_pcm_int16_le (OpenAI's response_format=pcm)
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("PCM int16 LE has no header", "[unit][pcm]") {
    std::vector<float> samples = {0.0f, 0.5f, -0.5f};
    std::string pcm = crispasr_make_pcm_int16_le(samples.data(), (int)samples.size());

    // Body must NOT start with RIFF — that would mean we accidentally
    // wrote a WAV header into the OpenAI pcm stream.
    REQUIRE(pcm.size() == 6); // 3 samples × 2 bytes
    REQUIRE(pcm.substr(0, 4) != "RIFF");
}

TEST_CASE("PCM body size = 2 × n_samples", "[unit][pcm]") {
    for (int n : {1, 17, 256, 24000, 100000}) {
        std::vector<float> samples(n, 0.5f);
        std::string pcm = crispasr_make_pcm_int16_le(samples.data(), n);
        REQUIRE(pcm.size() == (size_t)n * 2);
    }
}

TEST_CASE("PCM applies same clamp as WAV writer", "[unit][pcm]") {
    std::vector<float> samples = {0.0f, 1.5f, -1.5f, 2.0f, -2.0f, 1.0f, -1.0f};
    std::string pcm = crispasr_make_pcm_int16_le(samples.data(), (int)samples.size());

    REQUIRE(le_i16(pcm, 0) == 0);
    REQUIRE(le_i16(pcm, 2) == 32767);
    REQUIRE(le_i16(pcm, 4) == -32767);
    REQUIRE(le_i16(pcm, 6) == 32767);
    REQUIRE(le_i16(pcm, 8) == -32767);
    REQUIRE(le_i16(pcm, 10) == 32767);
    REQUIRE(le_i16(pcm, 12) == -32767);
}

TEST_CASE("PCM applies same rounding as WAV writer", "[unit][pcm]") {
    std::vector<float> samples = {0.5f, -0.5f, 0.00001f, 0.000031f};
    std::string pcm = crispasr_make_pcm_int16_le(samples.data(), (int)samples.size());

    REQUIRE(le_i16(pcm, 0) == 16384);
    REQUIRE(le_i16(pcm, 2) == -16384);
    REQUIRE(le_i16(pcm, 4) == 0);
    REQUIRE(le_i16(pcm, 6) == 1);
}

TEST_CASE("PCM handles empty input", "[unit][pcm]") {
    std::string pcm = crispasr_make_pcm_int16_le(nullptr, 0);
    REQUIRE(pcm.empty());

    // Negative is treated as zero, no pointer deref.
    std::string pcm_neg = crispasr_make_pcm_int16_le(nullptr, -7);
    REQUIRE(pcm_neg.empty());
}

TEST_CASE("PCM byte-stream matches WAV body byte-for-byte", "[unit][pcm]") {
    // Both serializers should write the same int16 LE samples — only the
    // 44-byte header differs. Verify by stripping the header.
    std::vector<float> samples = {0.0f, 0.25f, -0.25f, 0.5f, -0.5f, 0.99f, -0.99f, 1.0f, -1.0f};
    std::string wav = crispasr_make_wav_int16(samples.data(), (int)samples.size(), 24000);
    std::string pcm = crispasr_make_pcm_int16_le(samples.data(), (int)samples.size());
    REQUIRE(wav.substr(44) == pcm);
}
