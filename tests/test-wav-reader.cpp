#include <catch2/catch_test_macros.hpp>

#include "core/wav_reader.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace {
struct WavFixture {
    std::vector<uint8_t> bytes;
    std::filesystem::path path;

    WavFixture() {
        // Atomic directory creation also isolates separately discovered CTest
        // cases when several test processes run in parallel.
        for (unsigned i = 0; path.empty(); ++i) {
            auto dir = std::filesystem::path("crispasr_wav_chunks_tmp_" + std::to_string(i));
            if (std::filesystem::create_directory(dir))
                path = dir / "fixture.wav";
        }
        tag("RIFF");
        le32(0);
        tag("WAVE");
    }
    ~WavFixture() { std::filesystem::remove_all(path.parent_path()); }
    void tag(const char* text) { bytes.insert(bytes.end(), text, text + 4); }
    void le16(uint16_t n) {
        bytes.push_back(n & 255);
        bytes.push_back(n >> 8);
    }
    void le32(uint32_t n) {
        le16(n & 65535);
        le16(n >> 16);
    }
    void fmt(uint32_t size = 16, int channels = 1, int rate = 16000) {
        tag("fmt ");
        le32(size);
        le16(1);
        le16(channels);
        le32(rate);
        le32(rate * channels * 2);
        le16(channels * 2);
        le16(16);
        for (uint32_t i = 16; i < size; ++i)
            bytes.push_back(0);
        if (size & 1)
            bytes.push_back(0);
    }
    void extensible(int channels = 1, uint16_t valid_bits = 16, uint16_t cb_size = 22) {
        const size_t start = bytes.size();
        fmt(16, channels);
        bytes[start + 8] = 0xfe;
        bytes[start + 9] = 0xff;
        bytes[start + 4] = 40;
        le16(cb_size);
        le16(valid_bits);
        le32(channels == 1 ? 4 : (1u << channels) - 1); // speaker mask; averaging is independent of positions
        const uint8_t pcm_guid[16] = {1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};
        bytes.insert(bytes.end(), pcm_guid, pcm_guid + 16);
    }
    void metadata(const char* name, uint32_t size) {
        tag(name);
        le32(size);
        bytes.insert(bytes.end(), size, 0x55);
        if (size & 1)
            bytes.push_back(0);
    }
    void data(uint32_t declared_size = 4) {
        tag("data");
        le32(declared_size);
        le16(16384);
        le16(0xc000); // -16384
    }
    bool read(std::vector<float>& pcm, int& rate) {
        uint32_t size = bytes.size() - 8;
        for (int i = 0; i < 4; ++i)
            bytes[4 + i] = (size >> (8 * i)) & 255;
        FILE* f = std::fopen(path.string().c_str(), "wb");
        REQUIRE(f != nullptr);
        auto written = std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
        REQUIRE(written == bytes.size());
        return crispasr::core::read_wav_mono_pcm16(path.string(), pcm, rate);
    }
};
} // namespace

TEST_CASE("PCM16 reader traverses padded RIFF chunks", "[wav-reader][unit]") {
    WavFixture wav;
    SECTION("odd JUNK before format") {
        wav.metadata("JUNK", 1);
        wav.fmt();
    }
    SECTION("odd LIST after format") {
        wav.fmt();
        wav.metadata("LIST", 3);
    }
    SECTION("odd format extension") {
        wav.fmt(17);
    }
    SECTION("even metadata and extended format") {
        wav.metadata("JUNK", 2);
        wav.fmt(18);
    }
    wav.data();
    std::vector<float> pcm;
    int rate = 0;
    REQUIRE(wav.read(pcm, rate));
    CHECK(rate == 16000);
    REQUIRE(pcm.size() == 2);
    CHECK(pcm[0] == 0.5f);
    CHECK(pcm[1] == -0.5f);
}

TEST_CASE("PCM16 reader rejects malformed non-data chunks", "[wav-reader][unit]") {
    WavFixture wav;
    SECTION("format declares fewer than 16 bytes") {
        wav.fmt(15);
        wav.data();
    }
    SECTION("format declares zero bytes") {
        wav.fmt(0);
        wav.data();
    }
    SECTION("truncated format body") {
        wav.fmt();
        wav.bytes.resize(wav.bytes.size() - 1);
    }
    SECTION("truncated extension") {
        wav.fmt(18);
        wav.bytes.pop_back();
    }
    SECTION("missing odd format pad") {
        wav.fmt(17);
        wav.bytes.pop_back();
    }
    SECTION("missing odd chunk pad") {
        wav.metadata("JUNK", 1);
        wav.bytes.pop_back();
    }
    SECTION("oversized unknown payload") {
        wav.tag("JUNK");
        wav.le32(0x7fffffff);
    }
    SECTION("maximum odd unknown payload") {
        wav.tag("JUNK");
        wav.le32(0xffffffff);
    }
    SECTION("negative-looking unknown payload") {
        wav.tag("JUNK");
        wav.le32(0xfffffff8);
    }
    SECTION("invalid sample rate") {
        wav.fmt(16, 1, 0);
        wav.data();
    }
    std::vector<float> pcm;
    int rate = 0;
    CHECK_FALSE(wav.read(pcm, rate));
}

TEST_CASE("PCM16 reader preserves bounded partial data and channel averaging", "[wav-reader][unit]") {
    WavFixture wav;
    int channels = 1;
    uint32_t size = 4;
    SECTION("stereo average") {
        channels = 2;
    }
    SECTION("huge truncated data") {
        size = 0x7ffffffe;
    }
    SECTION("unsigned truncated data") {
        size = 0xfffffffe;
    }
    SECTION("partial stereo frame discarded") {
        channels = 2;
        size = 6;
    }
    SECTION("empty data") {
        size = 0;
    }
    wav.fmt(16, channels);
    wav.data(size);
    if (size == 6)
        wav.le16(32767);
    std::vector<float> pcm;
    int rate = 0;
    REQUIRE(wav.read(pcm, rate));
    CHECK(rate == 16000);
    if (size == 0) {
        CHECK(pcm.empty());
    } else if (channels == 2) {
        REQUIRE(pcm.size() == 1);
        CHECK(pcm[0] == 0.0f);
    } else {
        REQUIRE(pcm.size() == 2);
        CHECK(pcm[0] == 0.5f);
        CHECK(pcm[1] == -0.5f);
    }
}

TEST_CASE("PCM16 reader accepts extensible PCM with exact samples", "[wav-reader][unit]") {
    WavFixture wav;
    int channels = 1;
    SECTION("mono") {}
    SECTION("stereo") {
        channels = 2;
    }
    SECTION("three channels") {
        channels = 3;
    }
    SECTION("odd metadata around extensible format") {
        wav.metadata("JUNK", 1);
        wav.extensible();
        wav.metadata("LIST", 3);
    }
    SECTION("larger extension with odd padding") {
        wav.extensible(1, 16, 23);
        wav.bytes[16] = 41;
        wav.bytes.push_back(0x55);
        wav.bytes.push_back(0);
    }
    if (wav.bytes.size() == 12)
        wav.extensible(channels);
    wav.data(channels == 3 ? 6 : 4);
    if (channels == 3)
        wav.le16(24576);
    std::vector<float> pcm;
    int rate = 0;
    REQUIRE(wav.read(pcm, rate));
    CHECK(rate == 16000);
    if (channels > 1) {
        REQUIRE(pcm.size() == 1);
        CHECK(pcm[0] == (channels == 3 ? 0.25f : 0.0f));
    } else {
        REQUIRE(pcm.size() == 2);
        CHECK(pcm[0] == 0.5f);
        CHECK(pcm[1] == -0.5f);
    }
}

TEST_CASE("PCM16 reader rejects invalid extensible layouts", "[wav-reader][unit]") {
    WavFixture wav;
    wav.extensible();
    SECTION("float subformat") {
        wav.bytes[44] = 3;
    }
    SECTION("non-PCM GUID tail") {
        wav.bytes[59] ^= 1;
    }
    SECTION("non-PCM GUID high tag byte") {
        wav.bytes[45] = 1;
    }
    SECTION("short cbSize") {
        wav.bytes[36] = 21;
    }
    SECTION("cbSize exceeds chunk") {
        wav.bytes[36] = 23;
    }
    SECTION("short format chunk") {
        wav.bytes[16] = 39;
    }
    SECTION("truncated extension") {
        wav.bytes.pop_back();
    }
    SECTION("zero valid bits") {
        wav.bytes[38] = 0;
    }
    SECTION("too many valid bits") {
        wav.bytes[38] = 17;
    }
    SECTION("unsupported valid bits") {
        wav.bytes[38] = 12;
    }
    SECTION("unsupported sample container") {
        wav.bytes[34] = 24;
    }
    SECTION("incorrect block alignment") {
        wav.bytes[32] = 4;
    }
    wav.data();
    std::vector<float> pcm;
    int rate = 0;
    CHECK_FALSE(wav.read(pcm, rate));
}
