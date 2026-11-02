// test-mossformergan-se.cpp — MossFormerGAN_SE_16K enhancement through the C ABI.
//
// [unit]  model-free: the enhancer table advertises "mossformergan-se", and a
//         metadata-only GGUF with that architecture (no tensors) is rejected
//         cleanly instead of crashing.
// [live]  MOSSFORMERGAN_SE_MODEL=<gguf>: enhance samples/jfk.wav + white noise
//         (11 s, i.e. the segmented > 10 s decode path) and a 3 s crop, check
//         length, finiteness and a real SNR gain; 48 kHz input, a sub-frame
//         input and digital silence keep the same-length contract.
//         Without the variable the live cases SKIP, never pass.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Minimal RIFF/WAVE PCM16 mono reader (samples/jfk.wav).
bool read_wav16(const std::string& path, std::vector<float>& out, int& sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    char riff[12];
    f.read(riff, 12);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(riff + 8, "WAVE", 4) != 0)
        return false;
    int channels = 0, bits = 0;
    while (f) {
        char id[4];
        uint32_t size = 0;
        f.read(id, 4);
        f.read((char*)&size, 4);
        if (!f)
            return false;
        if (!std::memcmp(id, "fmt ", 4)) {
            std::vector<char> b(size);
            f.read(b.data(), size);
            channels = *(uint16_t*)&b[2];
            sr = (int)*(uint32_t*)&b[4];
            bits = *(uint16_t*)&b[14];
        } else if (!std::memcmp(id, "data", 4)) {
            if (channels != 1 || bits != 16)
                return false;
            std::vector<int16_t> pcm(size / 2);
            f.read((char*)pcm.data(), size);
            out.resize(pcm.size());
            for (size_t i = 0; i < pcm.size(); i++)
                out[i] = pcm[i] / 32768.0f;
            return true;
        } else {
            f.seekg(size + (size & 1), std::ios::cur);
        }
    }
    return false;
}

std::vector<float> add_noise(const std::vector<float>& x, float sigma) {
    std::vector<float> y(x.size());
    uint64_t s = 0x9E3779B97F4A7C15ull;
    auto uni = [&]() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return ((s >> 11) + 0.5) / 9007199254740992.0;
    };
    for (size_t i = 0; i < x.size(); i++) {
        const double g = std::sqrt(-2.0 * std::log(uni())) * std::cos(6.283185307179586 * uni());
        y[i] = x[i] + sigma * (float)g;
    }
    return y;
}

double snr_db(const std::vector<float>& ref, const std::vector<float>& est) {
    double s = 0, e = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        s += (double)ref[i] * ref[i];
        e += ((double)est[i] - ref[i]) * ((double)est[i] - ref[i]);
    }
    return 10.0 * std::log10(s / e);
}

const char* live_model() {
    const char* m = std::getenv("MOSSFORMERGAN_SE_MODEL");
    return (m && m[0]) ? m : nullptr;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x))
            return false;
    return true;
}

} // namespace

TEST_CASE("mossformergan-se is a registered enhancement architecture", "[unit][enhance][mossformergan-se]") {
    const char* archs = crispasr_enhance_model_archs();
    REQUIRE(archs != nullptr);
    CHECK(std::string(archs).find("mossformergan-se") != std::string::npos);
}

TEST_CASE("mossformergan-se rejects a GGUF without weights", "[unit][enhance][mossformergan-se]") {
    const std::string path = std::string(std::tmpnam(nullptr)) + "-mossformergan-se-stub.gguf";
    gguf_context* g = gguf_init_empty();
    REQUIRE(g != nullptr);
    gguf_set_val_str(g, "general.architecture", "mossformergan-se");
    gguf_set_val_u32(g, "mossformergan_se.layout", 3);
    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/true));
    gguf_free(g);
    std::vector<float> in(1600, 0.1f), out(1600, 0.0f);
    CHECK(crispasr_enhance_audio_spec(path.c_str(), in.data(), (int32_t)in.size(), 16000, 1, out.data(),
                                      (int32_t)out.size()) == -2);
    std::remove(path.c_str());
}

TEST_CASE("mossformergan-se live: segmented 11 s decode improves SNR", "[live][enhance][mossformergan-se]") {
    const char* model = live_model();
    if (!model)
        SKIP("MOSSFORMERGAN_SE_MODEL not set");
    std::vector<float> clean;
    int sr = 0;
    REQUIRE(read_wav16(std::string(CRISPASR_SOURCE_DIR) + "/samples/jfk.wav", clean, sr));
    REQUIRE(sr == 16000);
    REQUIRE(clean.size() > 160000); // > 10 s: exercises the upstream segmentation path
    const std::vector<float> noisy = add_noise(clean, 0.05f);
    std::vector<float> out(noisy.size());
    REQUIRE(crispasr_enhance_audio_spec(model, noisy.data(), (int32_t)noisy.size(), 16000, 4, out.data(),
                                        (int32_t)out.size()) == 0);
    REQUIRE(all_finite(out));
    const double before = snr_db(clean, noisy), after = snr_db(clean, out);
    INFO("SNR " << before << " dB -> " << after << " dB");
    CHECK(after > before + 5.0);
}

TEST_CASE("mossformergan-se live: same-length contract at other rates and edge sizes",
          "[live][enhance][mossformergan-se]") {
    const char* model = live_model();
    if (!model)
        SKIP("MOSSFORMERGAN_SE_MODEL not set");
    std::vector<float> clean;
    int sr = 0;
    REQUIRE(read_wav16(std::string(CRISPASR_SOURCE_DIR) + "/samples/jfk.wav", clean, sr));
    clean.resize(3 * 16000);
    std::vector<float> noisy = add_noise(clean, 0.03f);

    // 48 kHz in -> 48 kHz out (resampled to the native 16 kHz internally)
    std::vector<float> in48(noisy.size() * 3);
    for (size_t i = 0; i < in48.size(); i++)
        in48[i] = noisy[i / 3];
    std::vector<float> out48(in48.size());
    REQUIRE(crispasr_enhance_audio_spec(model, in48.data(), (int32_t)in48.size(), 48000, 4, out48.data(),
                                        (int32_t)out48.size()) == 0);
    CHECK(all_finite(out48));

    // fewer samples than one STFT half-window
    std::vector<float> tiny(150, 0.05f), tiny_out(150, 1.0f);
    REQUIRE(crispasr_enhance_audio_spec(model, tiny.data(), 150, 16000, 4, tiny_out.data(), 150) == 0);
    CHECK(all_finite(tiny_out));

    // digital silence stays silent (upstream's RMS normalisation would divide by zero)
    std::vector<float> zeros(8000, 0.0f), zeros_out(8000, 1.0f);
    REQUIRE(crispasr_enhance_audio_spec(model, zeros.data(), 8000, 16000, 4, zeros_out.data(), 8000) == 0);
    for (float v : zeros_out)
        REQUIRE(v == 0.0f);
}
