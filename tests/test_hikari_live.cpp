// Live acceptance coverage for hikari (sbintuitions/hikari-medium) through the
// session C ABI — the path the language bindings take.
//
// Required:
//   CRISPASR_MODEL_HIKARI=/path/to/hikari-medium-f16.gguf (or -q8_0)
// Silero VAD: ggml-silero-v6.2.0.bin next to the model (where -m auto puts
// it), or HIKARI_VAD_MODEL. Without it the policy's wait penalty never rises
// and translation comes back empty — which is what the German case guards:
// the C ABI used to open hikari without Silero and returned "" for en->de.
//
// Audio: samples/jfk.wav (16 kHz mono s16), from the repository root.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The reference implementation's output for jfk.wav, en->de (f16; q8_0 on CPU
// gives the same text).
constexpr const char* kGerman =
    "Und meine Mit-Amerikaner fragen nicht, was dein Land für dich tun kann. Frag, was du für dein Land tun kannst.";

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? v : "";
}

bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return false;
    std::fclose(f);
    return true;
}

// 16-bit PCM WAV, any chunk order.
std::vector<float> read_wav(const std::string& path) {
    std::vector<float> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return out;
    char riff[12];
    if (std::fread(riff, 1, 12, f) != 12 || std::memcmp(riff, "RIFF", 4) != 0) {
        std::fclose(f);
        return out;
    }
    char id[4];
    uint32_t size = 0;
    while (std::fread(id, 1, 4, f) == 4 && std::fread(&size, 4, 1, f) == 1) {
        if (std::memcmp(id, "data", 4) == 0) {
            std::vector<int16_t> pcm(size / 2);
            const size_t got = std::fread(pcm.data(), 2, pcm.size(), f);
            out.resize(got);
            for (size_t i = 0; i < got; i++)
                out[i] = pcm[i] / 32768.0f;
            break;
        }
        std::fseek(f, (long)size + (size & 1), SEEK_CUR);
    }
    std::fclose(f);
    return out;
}

std::string words(const std::string& s) {
    std::string w;
    bool space = false;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c >= 0x80) {
            if (space && !w.empty())
                w.push_back(' ');
            w.push_back((char)std::tolower(c));
            space = false;
        } else {
            space = true;
        }
    }
    return w;
}

std::string transcribe(crispasr_session* s, const std::vector<float>& pcm) {
    crispasr_session_result* r = crispasr_session_transcribe(s, pcm.data(), (int)pcm.size());
    if (!r)
        return "<null>";
    std::string text;
    for (int i = 0; i < crispasr_session_result_n_segments(r); i++) {
        if (!text.empty())
            text += ' ';
        text += crispasr_session_result_segment_text(r, i);
    }
    crispasr_session_result_free(r);
    return text;
}

} // namespace

TEST_CASE("hikari: English ASR and en->de through the session C ABI", "[hikari][live]") {
    const std::string model = env("CRISPASR_MODEL_HIKARI");
    if (model.empty() || !file_exists(model))
        SKIP("CRISPASR_MODEL_HIKARI not set or missing");
    const std::vector<float> pcm = read_wav("samples/jfk.wav");
    REQUIRE(pcm.size() == 176000);

    crispasr_session* s = crispasr_session_open(model.c_str(), 4);
    REQUIRE(s != nullptr);

    SECTION("English transcription") {
        REQUIRE(crispasr_session_set_target_language(s, "") == 0);
        const std::string text = transcribe(s, pcm);
        INFO(text);
        const std::string w = words(text);
        REQUIRE(w.find("fellow americans") != std::string::npos);
        REQUIRE(w.find("ask what you can do for your country") != std::string::npos);
    }
    SECTION("English to German equals the reference text") {
        REQUIRE(crispasr_session_set_target_language(s, "de") == 0);
        const std::string text = transcribe(s, pcm);
        INFO(text);
        REQUIRE(words(text) == words(kGerman));
    }
    crispasr_session_close(s);
}
